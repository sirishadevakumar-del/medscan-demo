
"""End-to-end inference: quality check -> modality routing -> ensemble + MC Dropout -> Grad-CAM -> decision."""

from pathlib import Path
import importlib.util

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms as T
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageOps

from .audit import AuditLog
from .data import build_transforms, load_any_image, to_rgb
from .gradcam import gradcam, overlay
from .modality import determine, tasks_for
from .models import load_checkpoint
from .policy import decide
from .quality import Thresholds, assess, metrics as raw_metrics
from .uncertainty import mc_predict
from .volume import cam_to_grid, select_slices


VOLUME_METHOD = (
    "Multi-slice aggregation of the existing 2D model "
    "(each axial slice is classified separately and the "
    "slice probabilities are averaged). This is NOT a trained 3D model."
)


class Analyzer:

    def __init__(self, ckpt, device=None, audit_path="runs/audit_log.jsonl"):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.ens, self.cfg, ck = load_checkpoint(ckpt, self.device)

        self.thresholds = {
            k: Thresholds(**v) for k, v in ck["thresholds"].items()
        }

        self.modalities = ck["modalities"]

        self.tasks = {
            k: v for k, v in ck["task_meta"].items()
            if k != "_modality"
        }

        self.tf = build_transforms(self.cfg.image_size, False)

        self.audit = AuditLog(audit_path)

        # --------------------------------------------------
        # Pretrained CT lung cancer model
        # --------------------------------------------------

        ct_path = hf_hub_download(
            repo_id="SirishaDevakumar/medscan-pretrained-models",
            filename="best_efficientnet_b0.pt",
            repo_type="model"
        )

        self.ct_classes = [
            "adenocarcinoma",
            "large.cell.carcinoma",
            "normal",
            "squamous.cell.carcinoma"
        ]

        self.ct_model = models.efficientnet_b0(weights=None)

        self.ct_model.classifier[1] = nn.Linear(1280, 4)

        self.ct_model.load_state_dict(
            torch.load(
                ct_path,
                map_location="cpu",
                weights_only=True
            ),
            strict=True
        )

        self.ct_model = self.ct_model.to(self.device).eval()

        self.ct_tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        # --------------------------------------------------
        # Pretrained MRI brain tumour model
        # --------------------------------------------------

        self.mri_folder = snapshot_download(
            repo_id="Purino/densenet121-brain-mri-tumor"
        )

        mri_inference_path = Path(self.mri_folder) / "inference.py"

        spec = importlib.util.spec_from_file_location(
            "medscan_mri_inference",
            str(mri_inference_path)
        )

        mri_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mri_module)

        self.mri_model, self.mri_config = mri_module.load_model(
            self.mri_folder
        )

        self.mri_model = self.mri_model.to(self.device).eval()

        self.mri_module = mri_module

    # --------------------------------------------------
    # CT prediction
    # --------------------------------------------------

    @torch.inference_mode()
    def predict_ct(self, image):

        """Predict lung cancer class from a CT image."""

        pil = (
            load_any_image(image)
            if isinstance(image, (str, Path))
            else image
        )

        pil = ImageOps.exif_transpose(pil).convert("RGB")

        x = self.ct_tf(pil).unsqueeze(0).to(self.device)

        probabilities = self.ct_model(x).softmax(1)[0].cpu()

        idx = int(probabilities.argmax())

        return {
            "task": "ct_lung_cancer",
            "modality": "ct",
            "prediction": self.ct_classes[idx],
            "confidence": float(probabilities[idx]),
            "probabilities": dict(zip(
                self.ct_classes,
                map(float, probabilities)
            ))
        }

    # --------------------------------------------------
    # MRI prediction
    # --------------------------------------------------

    @torch.inference_mode()
    def predict_mri(self, image):

        """Predict brain tumour class from an MRI image."""

        output = self.mri_module.predict(
            image,
            self.mri_model,
            self.mri_config
        )

        return {
            "task": "mri_brain_tumor",
            "modality": "mri",
            "prediction": output["label"],
            "confidence": output["confidence"],
            "probabilities": output["probabilities"]
        }

    # --------------------------------------------------
    # Modality routing
    # --------------------------------------------------

    def route(self, x):

        """Predict imaging modality, return (modality, probs)."""

        with torch.no_grad():
            p = self.ens(x, "_modality")[0].cpu().numpy()

        return self.modalities[int(p.argmax())], p

    # --------------------------------------------------
    # Modality detection
    # --------------------------------------------------

    def detect(
        self,
        image,
        dicom_tag=None,
        manual=None,
        ext_is_nifti=False,
        confirmed_mismatch=False
    ):

        """Combine classifier, DICOM metadata and the user's choice."""

        rgb = to_rgb(
            load_any_image(image)
            if isinstance(image, (str, Path))
            else image
        )

        guess, probs = self.route(
            self.tf(rgb).unsqueeze(0).to(self.device)
        )

        return determine(
            guess,
            float(probs.max()),
            dicom_tag,
            manual,
            ext_is_nifti,
            confirmed_mismatch,
            known=list(self.modalities)
        )

    # --------------------------------------------------
    # Image quality
    # --------------------------------------------------

    def quality_for(self, rgb, modality):

        """Run the scan-quality check."""

        ts = tasks_for(self.tasks, modality)

        if ts:
            return {
                **assess(rgb, self.thresholds[ts[0]]),
                "calibrated": True
            }

        m = raw_metrics(rgb)

        return {
            **m,
            "blur_ok": True,
            "contrast_ok": True,
            "artifact_ok": True,
            "passed": True,
            "score": float("nan"),
            "flags": [],
            "calibrated": False
        }

    # --------------------------------------------------
    # Precheck
    # --------------------------------------------------

    def precheck(self, image, task=None, modality=None):

        """Detect image type and run scan-quality check."""

        pil = (
            load_any_image(image)
            if isinstance(image, (str, Path))
            else image
        )

        rgb = to_rgb(pil)

        x = self.tf(rgb).unsqueeze(0).to(self.device)

        routed, mod_probs = self.route(x)

        modality = modality or routed

        if task is None:
            cands = tasks_for(self.tasks, modality)
            task = cands[0] if cands else None

        q = (
            assess(rgb, self.thresholds[task])
            if task
            else self.quality_for(rgb, modality)
        )

        return {
            "modality": routed,
            "modality_confidence": float(mod_probs.max()),
            "task": task,
            "quality": q
        }

    # --------------------------------------------------
    # Main analysis
    # --------------------------------------------------

    def analyze(
        self,
        image,
        task=None,
        patient_id=None,
        log=True,
        modality=None,
        mod_info=None
    ):

        pil = (
            load_any_image(image)
            if isinstance(image, (str, Path))
            else image
        )

        rgb = to_rgb(pil)

        x = self.tf(rgb).unsqueeze(0).to(self.device)

        routed_mod, probs_by_mod, disagree = None, {}, None

        if modality is None:

            routed_mod, mod_probs = self.route(x)

            probs_by_mod = dict(
                zip(self.modalities, map(float, mod_probs))
            )

            modality = routed_mod

            if task is not None:

                modality = self.tasks[task]["modality"]

                if routed_mod != modality:

                    disagree = (
                        f"The image classifier suggests '{routed_mod}' "
                        f"but task '{task}' is for '{modality}' images; "
                        "the result may not be meaningful."
                    )

        routed = None

        if task is None:

            cands = tasks_for(self.tasks, modality)

            if not cands:
                raise ValueError(
                    f"No task trained for modality '{modality}'"
                )

            task, routed = cands[0], cands

        meta = self.tasks[task]

        if (
            meta["modality"] != modality
            and (mod_info or {}).get("status") != "mismatch"
        ):

            raise ValueError(
                f"Task '{task}' is for {meta['modality']} images "
                f"but the image type is '{modality}'. "
                "Choose the matching image type."
            )

        classes = meta["classes"]

        q = assess(rgb, self.thresholds[task])

        mc = mc_predict(
            self.ens,
            x,
            task,
            self.cfg.mc_samples,
            self.cfg.confidence_mode
        )

        probs = mc["probs"][0].cpu().numpy()

        conf = float(mc["confidence"][0])

        if meta["multilabel"]:

            target = int(probs.argmax())

            pred_label = (
                [classes[i] for i in np.where(probs >= 0.5)[0]]
                or ["no finding above 0.5"]
            )

        else:

            target = int(mc["pred"][0])

            pred_label = classes[target]

        cam, _ = gradcam(self.ens, x, task, target)

        ov = overlay(
            np.asarray(
                rgb.resize((self.cfg.image_size,) * 2)
            ),
            cam
        )

        d = decide(
            conf,
            q["passed"],
            q["flags"],
            self.cfg.conf_high,
            self.cfg.conf_low
        )

        patient = self.audit.hash_id(patient_id)

        prior = self.audit.compare_with_prior(
            patient,
            task,
            probs.tolist()
        )

        res = {
            "modality": modality,
            "modality_probs": probs_by_mod,
            "modality_info": mod_info,
            "task": task,
            "candidate_tasks": routed,
            "prediction": pred_label,
            "probabilities": dict(zip(classes, map(float, probs))),
            "uncertainty_std": dict(zip(
                classes,
                map(float, mc["std"][0].cpu().numpy())
            )),
            "confidence": conf,
            "mc_confidence": float(mc["mc_confidence"][0]),
            "tier": d["tier"],
            "action": d["action"],
            "notes": d["notes"] + ([disagree] if disagree else []),
            "quality": q,
            "longitudinal": prior,
            "gradcam_target": classes[target],
            "multilabel": bool(meta["multilabel"])
        }

        if log:

            res["case_id"] = self.audit.log(
                patient=patient,
                task=task,
                pred_label=pred_label,
                probs=probs.tolist(),
                confidence=conf,
                tier=d["tier"],
                quality_score=q["score"]
            )

        return res, ov, cam

    # --------------------------------------------------
    # Analyze all tasks
    # --------------------------------------------------

    def analyze_all(
        self,
        image,
        patient_id=None,
        log=True,
        modality=None,
        mod_info=None
    ):

        """Run every trained task for the selected image type."""

        if modality is None:
            modality = self.precheck(image)["modality"]

        tasks = tasks_for(self.tasks, modality)

        return [
            self.analyze(
                image,
                t,
                patient_id,
                log,
                modality,
                mod_info
            )
            for t in tasks
        ]

    # --------------------------------------------------
    # Volume quality
    # --------------------------------------------------

    def volume_quality(self, vol, modality, n_max=12):

        """Scan quality over a sample of axial slices."""

        idx = select_slices(vol, n_max)

        qs = [
            self.quality_for(vol.axial_pil(k), modality)
            for k in idx
        ]

        flags = {}

        for q in qs:
            for f in q["flags"]:
                flags[f] = flags.get(f, 0) + 1

        frac = float(
            np.mean([q["passed"] for q in qs])
        )

        agg = (
            {
                k: float(np.nanmean([q[k] for q in qs]))
                for k in ("blur", "contrast", "edge_density", "score")
            }
            if qs[0]["calibrated"]
            else {
                k: float(np.mean([q[k] for q in qs]))
                for k in ("blur", "contrast", "edge_density")
            }
        )

        agg.setdefault("score", float("nan"))

        return {
            **agg,
            "passed": frac >= 0.5,
            "fraction_passed": frac,
            "calibrated": qs[0]["calibrated"],
            "flags": [
                f for f, n in flags.items()
                if n >= len(qs) / 2
            ],
            "blur_ok": True,
            "contrast_ok": True,
            "artifact_ok": True
        }

    # --------------------------------------------------
    # Multi-slice volume analysis
    # --------------------------------------------------

    def analyze_volume(
        self,
        vol,
        modality,
        patient_id=None,
        log=True,
        n_max=24,
        mod_info=None
    ):

        """Multi-slice analysis using the existing 2D ensemble."""

        idx = select_slices(vol, n_max)

        out = []

        for task in tasks_for(self.tasks, modality):

            meta = self.tasks[task]

            classes = meta["classes"]
            ml = bool(meta["multilabel"])

            per, ovs, cams, grid, std = [], {}, {}, {}, []

            for k in idx:

                res, ov, cam = self.analyze(
                    vol.axial_pil(k),
                    task,
                    None,
                    False,
                    modality,
                    mod_info
                )

                P = np.array([
                    res["probabilities"][c]
                    for c in classes
                ])

                per.append({
                    "z": k,
                    "prediction": res["prediction"],
                    "probabilities": P.tolist(),
                    "confidence": res["confidence"],
                    "tier": res["tier"],
                    "quality_passed": res["quality"]["passed"],
                    "quality": res["quality"]
                })

                std.append([
                    res["uncertainty_std"][c]
                    for c in classes
                ])

                ovs[k] = ov
                cams[k] = cam
                grid[k] = cam_to_grid(cam, vol)

            P = np.array([
                p["probabilities"]
                for p in per
            ])

            mean = P.mean(0)

            top = int(mean.argmax())

            if ml:

                pred = (
                    [classes[i] for i in np.where(mean >= 0.5)[0]]
                    or ["no finding above 0.5"]
                )

                agree = float(
                    np.mean(
                        (P[:, top] >= 0.5)
                        == (mean[top] >= 0.5)
                    )
                )

                agreeing = [
                    p["confidence"]
                    for p, pr in zip(per, P)
                    if (pr[top] >= 0.5) == (mean[top] >= 0.5)
                ]

            else:

                pred = classes[top]

                votes = P.argmax(1)

                agree = float((votes == top).mean())

                agreeing = [
                    p["confidence"]
                    for p, v in zip(per, votes)
                    if v == top
                ]

            conf = float(
                agree * (np.mean(agreeing) if agreeing else 0.0)
            )

            qf = float(
                np.mean([p["quality_passed"] for p in per])
            )

            qpass = qf >= 0.5

            flags = sorted({
                f
                for p in per
                for f in p["quality"]["flags"]
            })

            d = decide(
                conf,
                qpass,
                flags,
                self.cfg.conf_high,
                self.cfg.conf_low
            )

            q = {
                k: float(np.mean([
                    p["quality"][k] for p in per
                ]))
                for k in ("blur", "contrast", "edge_density", "score")
            }

            q.update(
                passed=qpass,
                flags=flags,
                fraction_passed=qf,
                calibrated=True
            )

            patient = self.audit.hash_id(patient_id)

            res = {
                "modality": modality,
                "modality_info": mod_info,
                "task": task,
                "analysis_type": "multi-slice",
                "method": VOLUME_METHOD,
                "prediction": pred,
                "probabilities": dict(zip(classes, map(float, mean))),
                "uncertainty_std": dict(zip(
                    classes,
                    map(float, np.mean(std, 0))
                )),
                "confidence": conf,
                "slice_agreement": agree,
                "tier": d["tier"],
                "action": d["action"],
                "notes": d["notes"],
                "quality": q,
                "slices": per,
                "slice_indices": idx,
                "n_slices_total": int(vol.n_slices),
                "gradcam_target": classes[top],
                "multilabel": ml,
                "classes": classes,
                "longitudinal": self.audit.compare_with_prior(
                    patient,
                    task,
                    mean.tolist()
                ),
                "volume": vol.describe()
            }

            if log:

                res["case_id"] = self.audit.log(
                    patient=patient,
                    task=task,
                    pred_label=pred,
                    probs=mean.tolist(),
                    confidence=conf,
                    tier=d["tier"],
                    quality_score=q["score"],
                    analysis="multi-slice",
                    n_slices=len(idx)
                )

            out.append((res, ovs, grid))

        return out


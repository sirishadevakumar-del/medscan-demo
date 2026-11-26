"""The Streamlit page itself, shared by the local app and the public demo.

Sections: 1 role | 2 upload | 3 image-type detection | 4 scan quality | 5 Analyze anyway | 6 AI findings (+Grad-CAM)
| 7 multi-slice CT/MRI | 8 interactive 3D viewer | 9 progression | 10 report | 11 feedback and audit history.

The user never chooses a disease or task. The image type is detected (or confirmed by the user) and every trained
analysis for that type is run. The role only changes how results are shown, never the results themselves.
"""
import shutil
import tempfile

import cv2
from pathlib import Path

import numpy as np

from . import progression as prog
from .modality import ALL_MODALITIES, label, tasks_for
from .report import build_report, to_json, to_pdf, to_text
from .view import ROLES, banners, modality_status_text, render_result, render_summary
from . import theme
from .viz3d import (area_curve_figure, blend, decorate, probability_comparison_figure, slice_curve_figure, volume_figure)
from .volume import attention_volume, load_mask, load_study, mask_stats

UPLOAD_TYPES = ["png", "jpg", "jpeg", "dcm", "dicom", "nii", "gz", "zip"]
AUTO = "Auto-detect (recommended)"


# ------------------------------------------------------------------------------------------------ helpers
def _load_uploads(files):
    d = tempfile.mkdtemp()
    try:
        paths = []
        for i, f in enumerate(files):
            p = Path(d) / f"{i}_{Path(f.name).name}"
            p.write_bytes(f.getvalue())
            paths.append(p)
        return load_study(paths)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _load_mask_upload(f, vol):
    d = tempfile.mkdtemp()
    try:
        p = Path(d) / Path(f.name).name
        p.write_bytes(f.getvalue())
        return load_mask(p, vol)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _det_image(study):
    return study["image"] if study["kind"] == "image" else study["volume"].axial_pil(study["volume"].n_slices // 2)


def _brightness(pil):
    return float(np.asarray(pil.convert("L")).mean())


def _reset(st):
    for k in [k for k in st.session_state if str(k).startswith("mk_") and k != "mk_key"]:
        del st.session_state[k]


# ------------------------------------------------------------------------------------------------ main
def render_app(st, an, samples=None, patient_id_field=True, metrics=None, eval_dir=None):
    samples = samples or {}
    theme.inject(st)
    theme.hero(st)
    # 1-2. sidebar: role + upload -------------------------------------------------------------------------------
    with st.sidebar:
        theme.brand(st)
        st.markdown("##### 1. Viewer")
        role = st.selectbox("Who is viewing this result?", ROLES)
        st.markdown("##### 2. Medical scan upload")
        ups = st.file_uploader("X-ray, CT, MRI, ultrasound, skin, eye or tissue image. CT/MRI volume: NIfTI (.nii/.nii.gz), "
                               "or DICOM series (several .dcm files or one .zip).",
                               type=UPLOAD_TYPES, accept_multiple_files=True)
        choice = st.selectbox("...or try a sample image", ["(none)"] + sorted(samples)) if samples else "(none)"
        pid = st.text_input("Patient ID (optional, stored hashed)") if patient_id_field else ""
    if ups:
        src = ("upload", tuple((u.name, u.size) for u in ups))
    elif choice != "(none)":
        src = ("sample", choice)
    else:
        st.info("Upload a scan in the sidebar (or pick a sample) to begin.")
        theme.feature_strip(st)
        return
    key = (src, pid)
    if st.session_state.get("mk_key") != key:
        _reset(st)
        st.session_state["mk_key"] = key
    kid = str(abs(hash(key)))
    try:
        if "mk_study" not in st.session_state:
            with st.spinner("Reading scan..."):
                st.session_state["mk_study"] = _load_uploads(ups) if ups else load_study([samples[choice]])
        study = st.session_state["mk_study"]
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        return
    is_vol, vol = study["kind"] == "volume", study["volume"]
    det_img = _det_image(study)

    # 3. image type (sidebar selector, messages in the page) ------------------------------------------------------
    with st.sidebar:
        st.markdown("##### 3. Image type")
        pick = st.selectbox("Image type (leave on auto-detect unless the detected type is wrong)",
                            [AUTO] + [label(m) for m in ALL_MODALITIES], key="type_" + kid)
        n_max = 24
        if is_vol:
            n_max = st.slider("Axial slices to analyse", 8, 48, 24, 4, key="nmax_" + kid)
        st.markdown('<div class="ms-card"><h4>About</h4><span class="ms-mut">Decision support only. The clinician makes '
                    'the final decision. Not for clinical use. Do not upload identifiable patient data to a public demo.'
                    '</span></div>', unsafe_allow_html=True)
    manual = None if pick == AUTO else next(m for m in ALL_MODALITIES if label(m) == pick)
    mod = an.detect(det_img, study["dicom_modality"], manual, study["is_nifti"], False)
    if mod.status == "mismatch":
        for w in mod.warnings:
            st.error(w)
        if st.checkbox("I understand the mismatch and want to analyze with the type I selected "
                       "(result will be marked as a modality mismatch)", key="mm_" + kid):
            mod = an.detect(det_img, study["dicom_modality"], manual, study["is_nifti"], True)
    elif mod.status == "unverified":
        for w in mod.warnings:
            st.warning(w)
        g = mod.classifier_guess
        if g and tasks_for(an.tasks, g) and st.checkbox(
                f"Analyze anyway as {label(g)} (result will be marked modality-UNVERIFIED and may not be meaningful)",
                key="un_" + kid):
            mod.value, mod.blocked = g, False
    if mod.status not in ("mismatch", "unverified"):
        for w in mod.warnings:
            st.caption(w)

    # 4-5. scan quality + analyze anyway ---------------------------------------------------------------------------
    q = an.volume_quality(vol, mod.value) if is_vol else an.quality_for(det_img.convert("RGB"), mod.value)
    quality_ok = q["passed"]
    if not q["passed"]:
        st.warning("Scan-quality check flagged this image: " + ", ".join(q["flags"] or ["low quality"]) + ".")
        quality_ok = st.checkbox("Analyze anyway", key="anyway_" + kid)
        if not quality_ok:
            st.caption("Analysis is paused until you tick 'Analyze anyway'. This only overrides the quality warning, "
                       "not an image-type problem.")
    can_run = bool(mod.value) and not mod.blocked and quality_ok

    # status strip ---------------------------------------------------------------------------------------------------
    vcol = {"verified": "#22c55e", "detected": "#3b82f6", "manual": "#f59e0b", "mismatch": "#ef4444", "unverified": "#ef4444"}
    tstr = (modality_status_text(mod.to_dict(), mod.value) if mod.value else
            f"not established (guess: {label(mod.classifier_guess)}, {mod.classifier_confidence:.0%})")
    theme.chips(st, [
        ("Image type", f"{label(mod.value) if mod.value else 'Unknown'} " + theme.pill(mod.status, vcol.get(mod.status, "#8da2c0"))),
        ("Scan", (f"Volume {vol.shape[0]}x{vol.shape[1]}x{vol.shape[2]}" if is_vol else f"2D image {det_img.size[0]}x{det_img.size[1]}")),
        ("Quality", theme.pill("Good" if q["passed"] else "Flagged", "#22c55e" if q["passed"] else "#f59e0b")),
        ("Viewing as", role)])
    st.caption("Image type: " + tstr)
    qc = st.columns(4)
    qc[0].metric("Blur score", f"{q['blur']:.0f}")
    qc[1].metric("Brightness (0-255)", f"{_brightness(det_img):.0f}")
    qc[2].metric("Contrast", f"{q['contrast']:.0f}")
    qc[3].metric("Quality status", "Good" if q["passed"] else "Flagged")
    if not q.get("calibrated", True):
        st.caption("No calibrated quality thresholds exist for this image type: raw metrics are shown, not judged.")
    if is_vol:
        st.caption(f"Voxel spacing {vol.spacing[0]:.2f}x{vol.spacing[1]:.2f}x{vol.spacing[2]:.2f} mm, orientation {vol.orientation}.")
    st.caption("Supported analysis types: " + ", ".join(label(m) for m in sorted({v['modality'] for v in an.tasks.values()}))
               + ". Other types (for example MRI) can be viewed but have no trained classifier.")

       # 6-8. analysis -----------------------------------------------------------------------------------------------
    runs = []
    extra_results = []

    akey = (mod.value, mod.status, bool(quality_ok), n_max, is_vol)

    has_original = bool(tasks_for(an.tasks, mod.value))
    has_standalone = not is_vol and mod.value == "ct"

    if can_run and (has_original or has_standalone):
        if st.session_state.get("mk_akey") != akey:
            with st.spinner("Analysing..." if not is_vol else "Analysing slices..."):
                if is_vol:
                    out = an.analyze_volume(
                        vol, mod.value, pid or None, True, n_max, mod.to_dict()
                    )
                    st.session_state["mk_res"] = [
                        {"res": r, "ovs": o, "grid": g} for r, o, g in out
                    ]
                else:
                    out = an.analyze_all(
                        det_img, pid or None, True, mod.value, mod.to_dict()
                    )
                    st.session_state["mk_res"] = [
                        {"res": r, "ov": o, "cam": c} for r, o, c in out
                    ]

                standalone = []
                if has_standalone:
                    standalone.append(an.predict_ct(det_img))

                st.session_state["mk_extra_res"] = standalone

            st.session_state["mk_akey"] = akey

        runs = st.session_state.get("mk_res", [])
        extra_results = st.session_state.get("mk_extra_res", [])

    elif st.session_state.get("mk_akey") is not None:
        st.session_state.pop("mk_akey", None)
        st.session_state.pop("mk_res", None)
        st.session_state.pop("mk_extra_res", None)

    if can_run and not has_original and not has_standalone:
        st.info(
            f"No classifier is available for {label(mod.value)} images. "
            "The viewer, quality check and report are still available."
        )

    if not can_run:
        st.warning("AI analysis has not run yet: " + (
            "confirm the image type in the sidebar."
            if not mod.value or mod.blocked
            else "tick 'Analyze anyway' above."
        ))

    results = [r["res"] for r in runs]

    names = ["AI findings"] + (["Multi-slice analysis", "3D viewer"] if is_vol else []) + \
            ["Progression", "Report", "Feedback & audit"]
    tabs = dict(zip(names, st.tabs(names)))

    
    with tabs["AI findings"]:
        if not results and not extra_results:
            st.caption("No AI findings to show yet.")
        else:
            if results:
                render_summary(st, [(r, None, None) for r in results], role)
                orig = np.asarray(det_img.convert("RGB"))
                for r in runs:
                    if "ov" in r:
                        ov = r["ov"]
                    else:
                        zc = min(r["ovs"], key=lambda z: abs(z - vol.n_slices // 2))
                        ov, orig = r["ovs"][zc], vol.slice2d("axial", zc)
                    with st.container(border=True):
                        render_result(st, r["res"], ov, role, metrics, orig=orig, cam=r.get("cam"), eval_dir=eval_dir)

            for extra in extra_results:
                st.markdown("---")
                st.subheader("CT Lung Cancer Classification")
                c1, c2 = st.columns(2)
                c1.metric("Predicted class", extra["prediction"])
                c2.metric("Model confidence", f"{extra['confidence']:.2%}")

                st.markdown("**Class-wise prediction scores**")
                probability_rows = [
                    {"Class": name, "Score": f"{score:.2%}"}
                    for name, score in extra["probabilities"].items()
                ]
                st.dataframe(
                    probability_rows,
                    hide_index=True,
                    use_container_width=True
                )

                st.caption(
                    "Standalone pretrained model. Scores are not calibrated clinical "
                    "probabilities. This is research/educational decision support only; "
                    "a qualified clinician must review the scan."
                )


    if is_vol:
        with tabs["Multi-slice analysis"]:
            _multislice(st, runs, vol, role, kid)
        with tabs["3D viewer"]:
            mask = _viewer(st, runs, vol, role, kid)
    else:
        mask = None

    with tabs["Progression"]:
        cmp_summary = _progression(st, an, study, det_img, mod, runs, mask, role, kid, pid)

    mask_measure = None
    if mask is not None:
        ms = mask_stats(mask, vol.spacing)
        mask_measure = {"volume_ml": ms["volume_ml"], "max_axial_area_mm2": ms["max_axial_area_mm2"],
                        "note": "Measured from a mask supplied by the user; this system did not create or validate it."}
    with tabs["Report"]:
        _report(st, an, results, runs, role, mod, vol if is_vol else None, cmp_summary, mask_measure, pid, kid)

    with tabs["Feedback & audit"]:
        _feedback(st, an, results, role, pid, kid)
    theme.feature_strip(st)


# ------------------------------------------------------------------------------------------------ tabs
def _multislice(st, runs, vol, role, kid):
    st.caption("Three different things are shown here and must not be confused: (1) individual 2D slice predictions, "
               "(2) the multi-slice aggregation that combines them into a volume-level summary, and (3) true 3D "
               "deep-learning inference, which this system does NOT do.")
    if not runs:
        st.info("Run an analysis first (see the messages above). MRI has no trained classifier; use the 3D viewer tab.")
        return
    for r in runs:
        res, ovs = r["res"], r["ovs"]
        st.subheader(f"{res['task']}: volume-level summary")
        c = st.columns(4)
        c[0].metric("Volume prediction", ", ".join(res["prediction"]) if isinstance(res["prediction"], list)
                    else str(res["prediction"]))
        c[1].metric("Volume confidence", f"{res['confidence']:.0%}")
        c[2].metric("Slice agreement", f"{res['slice_agreement']:.0%}")
        c[3].metric("Slices analysed", f"{len(res['slice_indices'])} of {res['n_slices_total']}")
        st.caption(res["method"])
        if role == "Patient":
            continue
        classes = res["classes"]
        top = classes.index(res["gradcam_target"])
        st.plotly_chart(slice_curve_figure(res["slices"], classes, top), use_container_width=True, key=f"sc_{kid}_{res['task']}")
        st.dataframe([{"slice": s["z"], "prediction": s["prediction"] if not isinstance(s["prediction"], list)
                       else ", ".join(s["prediction"]), "confidence": round(s["confidence"], 3), "tier": s["tier"],
                       "quality ok": s["quality_passed"]} for s in res["slices"]], use_container_width=True)
        z = st.select_slider("Inspect an analysed slice", res["slice_indices"], value=res["slice_indices"][len(res["slice_indices"]) // 2],
                             key=f"zs_{kid}_{res['task']}")
        a, b = st.columns(2)
        a.image(vol.slice2d("axial", z), caption=f"Axial slice {z}", clamp=True)
        b.image(ovs[z], caption=f"Grad-CAM (this slice's top class), slice {z}")


def _viewer(st, runs, vol, role, kid):
    st.caption("Multi-planar viewer in radiological orientation (patient's right on the left of the screen) with the original "
               "voxel spacing. Overlays come only from (a) a segmentation mask that you upload, or (b) the Grad-CAM attention "
               "region, which shows where the 2D model looked and is NOT a segmentation. No segmentation model is used.")
    f = st.file_uploader("Optional: external segmentation mask (NIfTI, same grid as the volume)",
                         type=["nii", "gz"], key="mask_" + kid)
    mask = None
    if f is None:
        st.session_state.pop("mk_mask", None)
    else:
        try:
            mask = _load_mask_upload(f, vol)
            st.session_state["mk_mask"] = mask
            st.success(f"Mask loaded: {int(mask.sum()):,} voxels.")
        except Exception as e:
            st.error(f"Mask not used: {e}")
    att, att_f = None, None
    if runs:
        tasks = [r["res"]["task"] for r in runs]
        sel = st.selectbox("Grad-CAM attention from analysis", tasks, key="attsel_" + kid) if len(tasks) > 1 else tasks[0]
        res0 = next(r for r in runs if r["res"]["task"] == sel)
        f2 = (max(1, vol.shape[0] // 128), max(1, vol.shape[1] // 128), 1)
        ck = ("att2d", sel)
        if st.session_state.get("mk_attkey") != ck:
            st.session_state["mk_att2d"] = attention_volume(vol, res0["grid"], f2)
            st.session_state["mk_attkey"] = ck
        att, att_f = st.session_state["mk_att2d"], f2
    opts = ["Original"] + (["Grad-CAM attention region (not a segmentation)"] if att is not None else []) + \
           (["External mask"] if mask is not None else []) + (["Both overlays"] if att is not None and mask is not None else [])
    mode = st.radio("View", opts, horizontal=True, key="vm_" + kid)
    use_m = mode in ("External mask", "Both overlays") and mask is not None
    use_a = mode in ("Grad-CAM attention region (not a segmentation)", "Both overlays") and att is not None

    X, Y, Z = vol.shape
    s1, s2, s3 = st.columns(3)
    iz = s1.slider("Axial slice", 0, Z - 1, Z // 2, key=f"sl_axial_{kid}")
    iy = s2.slider("Coronal slice", 0, Y - 1, Y // 2, key=f"sl_coronal_{kid}")
    ix = s3.slider("Sagittal slice", 0, X - 1, X // 2, key=f"sl_sagittal_{kid}")
    w1, w2 = st.columns(2)
    span = max(vol.vmax - vol.vmin, 1.0)
    wl0, ww0 = (vol.lo + vol.hi) / 2, vol.hi - vol.lo
    wl = w1.slider("Window level (WL)", float(vol.vmin), float(vol.vmax), float(min(max(wl0, vol.vmin), vol.vmax)),
                   step=float(span / 400), key=f"wl_{kid}")
    ww = w2.slider("Window width (WW)", 1.0, float(span), float(min(max(ww0, 1.0), span)), step=float(span / 400), key=f"ww_{kid}")
    lo, hi = wl - ww / 2, wl + ww / 2

    def panel(plane, idx):
        base = vol.slice2d(plane, idx, lo=lo, hi=hi)
        m2 = (vol.view(plane, idx, a=mask, nearest=True) > 0.5) if use_m else None
        h2 = None
        if use_a:
            ax_i = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
            ii = min(idx // att_f[ax_i], att.shape[ax_i] - 1)
            h2 = cv2.resize(vol.view(plane, ii, a=att, physical_aspect=False), (base.shape[1], base.shape[0]))
        img = blend(base, m2, h2) if (m2 is not None or h2 is not None) else np.repeat(base[..., None], 3, 2)
        return decorate(img, vol.letters(plane), vol.plane_coords(plane, ix, iy, iz))

    r1a, r1b = st.columns(2)
    r2a, r2b = st.columns(2)
    with r1a:
        st.markdown("**3D**")
        if st.checkbox("Render interactive 3D volume (rotate, zoom, pan)", value=True, key="r3d_" + kid):
            att3 = None
            if runs:
                _, _, f3 = vol.downsampled(56)
                att3 = attention_volume(vol, runs[0]["grid"], f3)
            st.plotly_chart(volume_figure(vol, mask, att3, show_mask=use_m, show_att=use_a, height=420),
                            use_container_width=True, key="p3d_" + kid)
        else:
            st.info("3D rendering is switched off.")
    r1b.markdown("**Axial**")
    r1b.image(panel("axial", iz), caption=f"Axial {iz + 1}/{Z}  |  WL {wl:.0f}  WW {ww:.0f}", use_container_width=True)
    r2a.markdown("**Sagittal**")
    r2a.image(panel("sagittal", ix), caption=f"Sagittal {ix + 1}/{X}", use_container_width=True)
    r2b.markdown("**Coronal**")
    r2b.image(panel("coronal", iy), caption=f"Coronal {iy + 1}/{Y}", use_container_width=True)
    st.caption("3D rendering uses a down-sampled copy for speed; the slice views use the full resolution. "
               "Blue lines mark where the other two planes cut.")
    with st.expander("Volume information (original data is never modified)"):
        st.json(vol.describe())
    return mask


def _progression(st, an, study, det_img, mod, runs, cur_mask, role, kid, pid):
    st.caption("Upload the PREVIOUS scan of the same patient to compare with the current one. " + prog.DISCLAIMER)
    pf = st.file_uploader("Previous scan (same image types as above)", type=UPLOAD_TYPES, accept_multiple_files=True,
                          key="prev_" + kid)
    d1, d2 = st.columns(2)
    pdate = d1.text_input("Previous scan date (optional, YYYY-MM-DD)", key="pd_" + kid)
    cdate = d2.text_input("Current scan date (optional, YYYY-MM-DD)", key="cd_" + kid)
    if not pf:
        if pid:
            st.caption("No previous scan uploaded. If this patient ID has an earlier analysis in the audit log, its "
                       "probability change is shown under AI findings.")
        else:
            st.info("No previous scan uploaded.")
        return None
    if not mod.value or mod.blocked:
        st.warning("Establish the current image type first.")
        return None
    pkey = ("prev", tuple((u.name, u.size) for u in pf), mod.value, bool(runs))
    if st.session_state.get("mk_pkey") != pkey:
        try:
            with st.spinner("Reading and analysing the previous scan..."):
                ps = _load_uploads(pf)
                pimg = _det_image(ps)
                pm = an.detect(pimg, ps["dicom_modality"], None, ps["is_nifti"], False)
                st.session_state["mk_prevstudy"] = (ps, pm)
                pres = []
                same_kind = ps["kind"] == study["kind"]
                compat = same_kind and not (pm.value and pm.status in ("verified", "detected") and pm.value != mod.value)
                if compat and tasks_for(an.tasks, mod.value):
                    pmd = {**pm.to_dict(), "value": mod.value}
                    if ps["kind"] == "volume":
                        pres = [{"res": r, "grid": g, "ovs": o} for r, o, g in
                                an.analyze_volume(ps["volume"], mod.value, None, False, 24, pmd)]
                    else:
                        pres = [{"res": r, "ov": o} for r, o, _ in an.analyze_all(pimg, None, False, mod.value, pmd)]
                st.session_state["mk_prevres"] = pres
            st.session_state["mk_pkey"] = pkey
        except Exception as e:
            st.error(f"Could not read the previous scan: {e}")
            return None
    ps, pm = st.session_state["mk_prevstudy"]
    pres = st.session_state["mk_prevres"]
    notes = []
    if ps["kind"] != study["kind"]:
        s = {"status": "Not comparable: one study is a single image, the other a volume.", "disclaimer": prog.DISCLAIMER}
        st.error(s["status"])
        return {**s, "previous_date": "not available", "current_date": "not available"}
    if pm.value and pm.status in ("verified", "detected") and pm.value != mod.value:
        s = {"status": f"Not comparable: previous study is {label(pm.value)}, current is {label(mod.value)}.",
             "disclaimer": prog.DISCLAIMER}
        st.error(s["status"])
        return {**s, "previous_date": "not available", "current_date": "not available"}
    if pm.status in ("unverified",) or not pm.value:
        notes.append("The previous study's image type could not be verified; it is assumed to match the current one.")
    # image comparison
    pimg = _det_image(ps)
    if study["kind"] == "volume":
        icmp = prog.compare_volumes(ps["volume"], study["volume"])
    else:
        icmp = prog.compare_images(pimg, det_img)
    # masks
    pmask = None
    if study["kind"] == "volume":
        mf = st.file_uploader("Optional: previous-study segmentation mask (NIfTI)", type=["nii", "gz"], key="pmask_" + kid)
        if mf is not None:
            try:
                pmask = _load_mask_upload(mf, ps["volume"])
            except Exception as e:
                st.error(f"Previous mask not used: {e}")
    mcmp = prog.compare_masks(pmask, cur_mask, ps["volume"].spacing if ps["volume"] else None,
                              study["volume"].spacing if study["volume"] else None) if study["kind"] == "volume" else None
    rows = prog.compare_predictions({r["res"]["task"]: r["res"] for r in pres}, {r["res"]["task"]: r["res"] for r in runs})
    summary = prog.summarise(mod.value, mod.value, rows, icmp, mcmp, pdate or ps["date"], cdate or study["date"], notes)
    (st.success if summary["status"] == "Comparable" else st.warning)("Comparison status: " + summary["status"])
    st.caption(f"Previous scan date: {summary['previous_date']}   |   Current scan date: {summary['current_date']}")
    if role == "Patient":
        st.write("Differences between the two scans are listed for your doctor. They do not say whether anything got better or worse.")
        return summary
    # timeline + baseline / current panels (like a radiology comparison viewer)
    st.markdown('<div class="ms-timeline">'
                f'<div class="ms-tl"><b>Baseline</b><small>{summary["previous_date"]} · {label(mod.value)}</small></div>'
                '<div style="color:#8da2c0">→</div>'
                f'<div class="ms-tl" style="border-left-color:#22c55e"><b>Current</b><small>{summary["current_date"]} · '
                f'{label(mod.value)}</small></div></div>', unsafe_allow_html=True)
    if study["kind"] == "volume":
        j = st.slider("Corresponding slice pair (matched by relative position)", 1, len(icmp["pairs"]), 3, key="pair_" + kid) - 1
        c = icmp["pairs"][j]
        pv, cv_ = ps["volume"], study["volume"]
        left, right = st.columns(2)
        for col, v, z, title in ((left, pv, c["prev_z"], "Baseline"), (right, cv_, c["curr_z"], "Current")):
            col.markdown(f"**{title}**  ·  axial slice {z + 1}/{v.n_slices}")
            im = np.repeat(v.slice2d("axial", z)[..., None], 3, 2)
            col.image(decorate(im, v.letters("axial")), use_container_width=True)
            tc = col.columns(len(icmp["pairs"]))
            for k, (t, pr) in enumerate(zip(tc, icmp["pairs"])):
                zz = pr["prev_z"] if title == "Baseline" else pr["curr_z"]
                th = np.repeat(v.slice2d("axial", zz)[..., None], 3, 2)
                if k == j:
                    th = cv2.copyMakeBorder(th, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(59, 130, 246))
                t.image(th, use_container_width=True)
    else:
        c = icmp
        left, right = st.columns(2)
        left.markdown("**Baseline**")
        left.image(c["previous"], use_container_width=True, clamp=True)
        right.markdown("**Current**")
        right.image(c["current"], use_container_width=True, clamp=True)
    # measurement + prediction cards under each panel
    ml_, mr_ = st.columns(2)
    pr_rows = [r for r in rows if r["status"] == "compared"]
    for col, side in ((ml_, "prev"), (mr_, "curr")):
        with col, st.container(border=True):
            st.markdown("**Segmentation measurement and predictions**")
            if mcmp:
                vol_ml = mcmp[f"{side}_volume_ml"]
                extra = "" if side == "prev" else (f' <span style="color:#8da2c0;font-size:1rem">'
                                                  f'({mcmp["delta_ml"]:+.2f} mL'
                                                  + ("" if mcmp["percent_change"] is None else f', {mcmp["percent_change"]:+.1f}%')
                                                  + ")</span>")
                st.markdown(f'<div style="font-size:2rem;font-weight:800">{vol_ml:.2f} <span style="font-size:1rem">mL</span>{extra}</div>',
                            unsafe_allow_html=True)
                st.caption("User-supplied mask, not validated by this system.")
            else:
                st.caption("Not available: needs a mask for BOTH studies.")
            for r_ in pr_rows:
                st.markdown(f"**{r_['task']}:** {r_[side + '_prediction']} ({r_[side + '_confidence']:.0%})")
    st.markdown("**1. Image differences** (alignment: " + icmp["alignment_status"] + ")")
    for n in icmp.get("notes", []) + notes:
        st.caption(n)
    cc = st.columns(3)
    for col, k, t in zip(cc, ("previous_aligned", "current", "difference"),
                         ("Baseline (aligned to current)", "Current", "Difference (bright = differs)")):
        col.image(c[k], caption=t, clamp=True, use_container_width=True)
    st.caption(f"Similarity (NCC) before/after alignment: {c['ncc_before']:.2f} / {c['ncc_after']:.2f}. Image differences "
               "can come from positioning or settings, not only from disease.")
    # predictions
    st.markdown("**2. Prediction differences** (model output, not a measurement of disease)")
    for row in rows:
        if row["status"] != "compared":
            st.caption(f"{row['task']}: {row['status']}")
            continue
        a, b = st.columns([2, 3])
        a.write({"previous": f"{row['prev_prediction']} ({row['prev_confidence']:.0%})",
                 "current": f"{row['curr_prediction']} ({row['curr_confidence']:.0%})",
                 "prediction changed": row["prediction_changed"]})
        b.plotly_chart(probability_comparison_figure(row["classes"], row["prev_probs"], row["curr_probs"], row["task"]),
                       use_container_width=True, key=f"pc_{kid}_{row['task']}")
    # segmentation
    st.markdown("**3. Validated segmentation measurements**")
    if mcmp:
        st.plotly_chart(area_curve_figure(mcmp["prev_curve"], mcmp["curr_curve"]), use_container_width=True, key="ac_" + kid)
        st.caption(mcmp["note"])
    else:
        st.caption("Not available: measured size needs a segmentation mask for BOTH studies (upload them in the 3D viewer "
                   "and above). The Grad-CAM attention region is never used for measurements.")
    with st.expander("Structured comparison summary (JSON)"):
        st.json(summary)
    return summary


def _report(st, an, results, runs, role, mod, vol, cmp_summary, mask_measure, pid, kid):
    rep = build_report(results, an.audit.hash_id(pid), role, mod.to_dict(), vol.describe() if vol else None,
                       cmp_summary, mask_measure)
    st.error(rep["banner"])
    left, right = st.columns([1, 2.4])
    with left, st.container(border=True):
        st.markdown("**Verified model outputs used**")
        st.markdown(f"**Patient (hashed):** {rep['patient_id_hashed']}  \n**Viewed as:** {rep['viewing_role']}  \n"
                    f"**Modality:** {rep['modality']} ({rep['modality_verification']})")
        for f in rep["findings"]:
            st.markdown(f"- {f['task_description']}: **{f['prediction']}** ({f['confidence_percent']}%, {f['confidence_category']})")
        if not rep["findings"]:
            st.caption("No AI prediction for this image type.")
        st.caption("The draft is built only from these outputs. Nothing else is inferred.")
    text = to_text(rep)
    with right:
        edited = st.text_area("Draft report (edit before use; every [VERIFY] item must be checked by a clinician)",
                              text, height=560, key=f"rt_{kid}_{abs(hash(text))}")
    imgs = []
    for r in runs:
        ov = r["ov"] if "ov" in r else r["ovs"][min(r["ovs"], key=lambda z: abs(z - vol.n_slices // 2))]
        imgs.append((f"Grad-CAM: {r['res']['task']}", ov))
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download PDF report", to_pdf(rep, imgs), "medscan_draft_report.pdf", "application/pdf", key="pdf_" + kid)
    c2.download_button("Download JSON report", to_json(rep), "medscan_draft_report.json", "application/json", key="json_" + kid)
    c3.download_button("Download edited text", edited, "medscan_draft_report.txt", "text/plain", key="txt_" + kid)
    st.caption("PDF and JSON contain the generated content; the .txt contains your edits.")


def _feedback(st, an, results, role, pid, kid):
    if role != "Patient" and results:
        st.subheader("Clinician feedback")
        agree = st.radio("Do you agree with the AI suggestions?", ["Yes", "No"], horizontal=True, key="ag_" + kid)
        final = st.text_input("Final diagnosis (if different)", key="fd_" + kid)
        note = st.text_area("Comment", key="nt_" + kid)
        if st.button("Submit feedback", key="fb_" + kid):
            for res in results:
                an.audit.feedback(res["case_id"], agree == "Yes", final or None, note)
            st.success("Feedback recorded in the audit log.")
    if pid:
        h = an.audit.hash_id(pid)
        rows = [r for r in an.audit._read() if r.get("patient") == h]
        st.subheader("Audit history for this patient ID")
        if rows:
            import time as _t
            st.dataframe([{"time": _t.strftime("%Y-%m-%d %H:%M", _t.localtime(r["time"])), "task": r["task"],
                           "prediction": r["pred_label"], "confidence": round(r["confidence"], 3), "tier": r["tier"],
                           "analysis": r.get("analysis", "2D")} for r in rows], use_container_width=True)
        else:
            st.caption("No earlier analyses stored for this patient ID.")

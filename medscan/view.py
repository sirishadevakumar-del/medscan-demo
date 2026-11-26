"""Role-aware result display, shared by the local app and the public demo.

`st` is the streamlit module, passed in so this file does not import it.
"""

ROLES = ["Doctor / Physician", "Radiologist", "Medical Student", "Researcher", "Patient"]
CLINICAL_ROLES = [r for r in ROLES if r != "Patient"]

# task -> (image type shown to people, what the model decides, is the answer a diagnosis?)
TASKS = {
    "xray_pneumonia": ("chest X-ray", "whether signs of pneumonia are present", True),
    "ct_organ": ("CT scan", "which organ is shown", False),
    "us_breast": ("breast ultrasound", "whether the finding looks malignant", True),
    "derm_lesion": ("skin (dermoscopy) image", "the type of skin lesion", True),
    "fundus_dr": ("retina (fundus) image", "the grade of diabetic retinopathy", True),
    "histo_colon": ("colon tissue (histopathology) slide", "the tissue type, including cancer tissue", True),
    "mri_brain_tumor": ("brain MRI", "the type of brain tumour, or none", True),
    "xray_chest14": ("chest X-ray", "which of 14 common chest findings are present", True),
    "xray_covid": ("chest X-ray", "COVID-19, pneumonia, lung opacity or normal lungs", True),
    "ct_chest_cancer": ("chest CT scan", "the type of lung cancer, or normal", True),
}

CLINICAL_TIER = {"high": "High-confidence finding.",
                 "medium": "Medium confidence: clinician review recommended.",
                 "low": "Low confidence: do not rely on this prediction; refer for expert review."}
PATIENT_TIER = {"high": "The AI is fairly sure about this result. Please still talk to your doctor.",
                "medium": "The AI is somewhat unsure. A doctor should review this image.",
                "low": "The AI is not sure. Please do not rely on this result and ask your doctor."}


def info(task):
    return TASKS.get(task, ("medical image", "the image class", True))


def task_label(task):
    return task if task.startswith("(") else f"{info(task)[0]} ({task})"


def _pred(res):
    p = res["prediction"]
    return ", ".join(p) if isinstance(p, list) else str(p)


def result_title(res):
    area, what, _ = info(res["task"])
    return f"{area.capitalize()}: {what}"


def render_summary(st, results, role):
    """One line per analysis that was run for the detected image type."""
    st.subheader("Summary")
    for res, _, _ in results:
        area, what, _ = info(res["task"])
        pred = _pred(res)
        if role == "Patient":
            st.markdown(f"- {what.capitalize()}: **{pred}** (AI certainty: {res['tier']})")
        else:
            st.markdown(f"- {what.capitalize()}: **{pred}** ({res['confidence']:.0%} confidence, {res['tier']} tier)")


def modality_status_text(mi, modality):
    """Plain description of how the image type was established (detected / manual / verified / unverified)."""
    from .modality import label
    if not mi:
        return f"{label(modality)} (not recorded)"
    st_, src = mi.get("status"), mi.get("source")
    if st_ == "verified":
        return f"{label(modality)}: VERIFIED from DICOM metadata"
    if st_ == "detected":
        return f"{label(modality)}: detected by the image classifier ({mi.get('classifier_confidence', 0):.0%} certain)"
    if st_ == "manual":
        return f"{label(modality)}: selected manually (not independently verified)"
    if st_ == "mismatch":
        return f"{label(modality)}: selected manually, CONFLICTS with the detected type"
    return f"{label(modality)}: UNVERIFIED"


def banners(st, res):
    """Warnings that stay visible next to every result: modality problems and low scan quality."""
    mi = res.get("modality_info") or {}
    if mi.get("status") == "mismatch":
        st.error("Modality/task mismatch was confirmed by the user. This prediction may not be meaningful.")
    elif mi.get("status") in ("unverified", "manual"):
        st.warning("Image type not independently verified: the prediction may not be meaningful if the type is wrong.")
    for w in mi.get("warnings", []):
        st.caption("Image type: " + w)
    if not res["quality"]["passed"]:
        st.warning("Based on a LOW-QUALITY scan (" + ", ".join(res["quality"]["flags"]) + "): interpret with extra caution.")


def render_result(st, res, ov, role, metrics=None, orig=None, cam=None, eval_dir=None):
    """One analysis result as a dashboard row: original | prediction card | Grad-CAM, then probabilities | performance."""
    from . import theme
    from .gradcam import heatmap
    area, what, is_dx = info(res["task"])
    pred, tier = _pred(res), res["tier"]
    patient = role == "Patient"
    st.markdown(f"#### {result_title(res)}")
    if res.get("analysis_type") == "multi-slice":
        st.caption(f"Volume-level result: {res['method']}")
    banners(st, res)

    def card(col):
        with col:
            theme.pred_card(st, "AI prediction", pred, res["confidence"], tier,
                            ("Tiers: high > 0.85, medium 0.70-0.85, low < 0.70" if not patient else ""), patient)
            {"high": st.success, "medium": st.warning, "low": st.error}[tier](
                (PATIENT_TIER if patient else CLINICAL_TIER)[tier])
            if res.get("multilabel"):
                top = sorted(res["probabilities"].items(), key=lambda kv: -kv[1])[:3]
                st.caption("Most likely findings: " + ", ".join(k if patient else f"{k} {v:.0%}" for k, v in top))
            if tier == "low":
                st.caption("No reliable finding: the image may show a condition this system was not trained on.")

    if patient:
        c1, c2 = st.columns(2)
        card(c1)
        with c2:
            st.write(f"The AI looked at your **{area}** and tried to decide **{what}**. Its answer: **{pred}**.")
            if not is_dx:
                st.info("This task recognises what is shown in the image. It is not a diagnosis.")
            if res["notes"]:
                st.warning("The picture quality may be too low for a reliable result. A clearer image could help.")
            with st.expander("Show where the AI looked"):
                st.image(ov, caption="Warmer colours: areas that influenced the AI most", width=400)
            st.caption("This is a research prototype and cannot replace a doctor.")
        return

    # --- clinical roles
    if role == "Doctor / Physician":
        c1, c2 = st.columns([1, 1])
        card(c1)
        with c1:
            _facts(st, res, area, what)
            _prior(st, res)
        c2.image(ov, caption=f"Grad-CAM for '{res['gradcam_target']}'", use_container_width=True)
        for n in res["notes"]:
            st.warning(n)
        with st.expander("Probabilities and model performance"):
            st.bar_chart(res["probabilities"])
            _performance(st, res, metrics, eval_dir)
        return
    c1, c2, c3 = st.columns([1, 1, 1.15])
    if orig is not None:
        c1.image(orig, caption="Original", use_container_width=True)
    card(c2)
    with c2:
        _facts(st, res, area, what)
    with c3:
        if cam is not None:
            h1, h2 = st.columns(2)
            h1.image(heatmap(cam), caption="Grad-CAM heat-map", use_container_width=True)
            h2.image(ov, caption="Overlay", use_container_width=True)
            st.caption("Low attention (blue) to high attention (red) for '" + res["gradcam_target"] + "'.")
        else:
            st.image(ov, caption=f"Grad-CAM overlay for '{res['gradcam_target']}'", use_container_width=True)
    for n in res["notes"]:
        st.warning(n)
    _prior(st, res)
    p1, p2 = st.columns([1.3, 1])
    with p1:
        st.markdown("**Prediction probabilities**")
        st.bar_chart(res["probabilities"])
    with p2:
        _performance(st, res, metrics, eval_dir)
    with st.expander("Uncertainty per class (standard deviation over Monte Carlo passes)", expanded=role == "Researcher"):
        st.write({k: round(v, 4) for k, v in res["uncertainty_std"].items()})
        st.write({"MC stability (1 - sigma/0.5)": round(res.get("mc_confidence", float("nan")), 4),
                  "combined confidence": round(res["confidence"], 4)})
    if role in ("Radiologist", "Researcher"):
        with st.expander("Technical image-analysis information", expanded=role == "Radiologist"):
            st.write({"quality metrics": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["quality"].items()
                                          if k != "flags"},
                      "image-type probabilities": {k: round(v, 3) for k, v in res.get("modality_probs", {}).items()}})
    if role == "Researcher" and metrics:
        with st.expander("Full evaluation record for this task (evaluate.py)"):
            st.json(metrics.get(res["task"], metrics))
    if role == "Medical Student":
        with st.expander("What is the confidence score?"):
            st.write("It is the probability of the predicted class multiplied by how stable that prediction is across "
                     "20 random dropout passes of two networks (Monte Carlo Dropout). Above 0.85 is high, "
                     "0.70 to 0.85 medium, below 0.70 low. It is not the chance of being correct.")
        with st.expander("What is Grad-CAM?"):
            st.write("The heat-map shows which image regions increased the score of the predicted class the most. "
                     "Red/yellow = influential. It shows where the model looked, not proof that it looked at the disease.")
        with st.expander("Why does the AI predict this?"):
            top = sorted(res["probabilities"].items(), key=lambda kv: -kv[1])[:2]
            st.write("The two highest model probabilities were " + " and ".join(f"'{k}' ({v:.0%})" for k, v in top) +
                     ". A prediction is only a statistical pattern match against the training images.")
        with st.expander("Why can the AI be wrong?"):
            st.write("It was trained on small public benchmark images (128 pixels). Images from other scanners, "
                     "other patients or unsupported image types can give unreliable results.")


def _performance(st, res, metrics, eval_dir):
    """Held-out test-set performance of the trained model for this task (from evaluate.py), never of this one image."""
    st.markdown("**Model performance** (benchmark test set)")
    m = ((metrics or {}).get(res["task"]) or {}).get("headline_mc_ensemble")
    if not m:
        st.caption("No evaluation results next to the checkpoint. Run evaluate.py to create runs/<exp>/eval/results.json.")
        return
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{m['accuracy']:.1%}")
    b.metric("AUC", f"{m['auc_macro']:.3f}")
    c.metric("Sensitivity*", f"{m['sensitivity_macro']:.1%}")
    st.caption("*macro average over classes. Measured on the benchmark's held-out test split, not on your image.")
    if eval_dir:
        from pathlib import Path
        roc = Path(eval_dir) / f"{res['task']}_roc.png"
        if roc.exists():
            st.image(str(roc), caption="ROC curve (test set)", use_container_width=True)


def _facts(st, res, area, what):
    q = res["quality"]
    st.markdown(f"**Modality:** {modality_status_text(res.get('modality_info'), res['modality'])}  \n"
                f"**Task:** {area}: {what} (`{res['task']}`)  \n"
                f"**Scan quality:** {'passed' if q['passed'] else 'FLAGGED: ' + ', '.join(q['flags'])} "
                f"(score {q['score']:.2f}, blur {q['blur']:.0f}, contrast {q['contrast']:.0f}, "
                f"edge density {q['edge_density']:.2f})")


def _prior(st, res):
    lo = res.get("longitudinal")
    if lo:
        st.write("Change vs previous study in the audit log (probability delta; not a measure of disease progression):",
                 lo["prob_change"])

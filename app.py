"""MedScan public demo (Streamlit Community Cloud)."""
import os
import tempfile

import streamlit as st

from medscan.pipeline import Analyzer
from medscan.webapp import render_app

# The trained model lives in a free Hugging Face *model* repository and is downloaded on first start.
# Change this to your own repository if you used a different name.
MODEL_REPO = os.environ.get("MEDSCAN_MODEL_REPO", "SirishaDevakumar/medscan-model")

st.set_page_config(page_title="MedScan demo", layout="wide")
st.warning("Research demo trained on small public datasets (128×128 images). It is not a medical device: "
           "never use it for real diagnosis or decisions, and do not upload real patient images.")


@st.cache_resource(show_spinner="Downloading the model (first start only)...")
def find_checkpoint():
    if os.path.exists("best.pt"):
        return "best.pt"
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=MODEL_REPO, filename="best.pt")


@st.cache_resource
def get_analyzer(path, version):
    return Analyzer(
        path,
        device="cpu",
        audit_path=os.path.join(
            tempfile.gettempdir(),
            "medscan_audit.jsonl"
        )
    )


try:
    an = get_analyzer(find_checkpoint(), "mri-enabled-v1")
except Exception as e:
    st.error(f"Could not get the model file from '{MODEL_REPO}': {e}")
    st.stop()

samples = {}
if os.path.isdir("samples"):
    for root, _, files in os.walk("samples"):
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                samples[os.path.relpath(os.path.join(root, f), "samples")] = os.path.join(root, f)

render_app(st, an, samples=samples, patient_id_field=False)

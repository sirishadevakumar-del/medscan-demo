"""Dark medical-dashboard look: CSS and small HTML components (cards, prediction card, status chips)."""
from html import escape

TIER_COLOR = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}

CSS = """
<style>
:root{--bg:#0b1220;--card:#111a2b;--card2:#0f1626;--line:#1f2d46;--blue:#3b82f6;--txt:#e6edf7;--mut:#8da2c0;}
.block-container{padding-top:1.2rem;max-width:1500px}
[data-testid="stSidebar"]{background:#0a1020;border-right:1px solid var(--line)}
[data-testid="stToolbar"],footer{visibility:hidden}
h1,h2,h3,h4{letter-spacing:.2px}
.ms-hero{text-align:center;padding:6px 0 14px}
.ms-hero h1{font-size:2.3rem;margin:0;font-weight:800}
.ms-hero h1 span{color:var(--blue)}
.ms-hero p{margin:2px 0;color:var(--mut)}
.ms-hero .tag{color:var(--blue);font-weight:600}
.ms-brand{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.25rem;margin-bottom:2px}
.ms-brand span{color:var(--blue)}
.ms-card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:12px}
.ms-card h4{margin:0 0 8px;font-size:.95rem;color:var(--txt)}
.ms-mut{color:var(--mut);font-size:.8rem}
.ms-chips{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 10px}
.ms-chip{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;min-width:150px}
.ms-chip b{display:block;font-size:.72rem;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.6px}
.ms-chip div{font-size:.95rem;font-weight:700;margin-top:2px}
.ms-pred{border:1px solid var(--line);border-radius:12px;padding:16px;background:var(--card2)}
.ms-pred .lab{font-size:.75rem;color:var(--mut);text-transform:uppercase;letter-spacing:.8px}
.ms-pred .val{font-size:1.9rem;font-weight:800;line-height:1.15;margin:4px 0 8px;word-break:break-word}
.ms-bar{height:9px;border-radius:6px;background:#1b2740;overflow:hidden;margin:6px 0}
.ms-bar i{display:block;height:100%;border-radius:6px}
.ms-pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:700;color:#0b1220}
.ms-feat{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0 6px}
.ms-feat div{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.ms-feat b{display:block;margin-bottom:4px;color:var(--blue)}
.ms-feat span{color:var(--mut);font-size:.82rem}
.ms-foot{text-align:center;color:var(--mut);font-size:.8rem;margin:10px 0 4px}
.ms-warnbar{background:#3a2a10;border:1px solid #6b4a12;border-radius:10px;padding:8px 12px;margin:6px 0;font-size:.88rem}
.ms-timeline{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:4px 0 10px}
.ms-tl{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--blue);border-radius:8px;padding:6px 12px}
.ms-tl small{color:var(--mut);display:block}
.ms-banner{background:#8b1021;color:#fff;border-radius:8px;padding:8px 14px;font-weight:800;text-align:center;letter-spacing:.4px;margin-bottom:10px}
@media(max-width:900px){.ms-feat{grid-template-columns:1fr 1fr}}
</style>
"""

LOGO = ('<svg width="30" height="30" viewBox="0 0 32 32" fill="none" stroke="#3b82f6" stroke-width="2.2" stroke-linecap="round">'
        '<rect x="3" y="3" width="26" height="26" rx="6"/><path d="M16 9v14M9 16h14"/></svg>')


def inject(st):
    st.markdown(CSS, unsafe_allow_html=True)


def hero(st):
    st.markdown('<div class="ms-hero"><h1>Med<span>Scan</span> AI</h1>'
                '<p>Adaptive, uncertainty-aware analysis for X-ray, CT, MRI, ultrasound, skin, eye and tissue images</p>'
                '<p class="tag">Deep learning. Explainable predictions. Interactive insights.</p></div>', unsafe_allow_html=True)


def brand(st):
    st.markdown(f'<div class="ms-brand">{LOGO}<div>Med<span>Scan</span> AI</div></div>'
                '<div class="ms-mut">Decision support for clinicians. Research prototype.</div>', unsafe_allow_html=True)


def chips(st, items):
    """items: list of (title, value). Status strip such as Image type | Verification | Quality | Analysis."""
    st.markdown('<div class="ms-chips">' + "".join(
        f'<div class="ms-chip"><b>{escape(t)}</b><div>{v}</div></div>' for t, v in items) + "</div>", unsafe_allow_html=True)


def pill(text, color):
    return f'<span class="ms-pill" style="background:{color}">{escape(text)}</span>'


def pred_card(st, title, label, conf, tier, subtitle="", patient=False):
    col = TIER_COLOR[tier]
    pct = "" if patient else f'<div style="font-weight:700">{conf:.1%} <span class="ms-mut">confidence</span></div>'
    bar = "" if patient else f'<div class="ms-bar"><i style="width:{max(2, conf * 100):.0f}%;background:{col}"></i></div>'
    st.markdown(f'<div class="ms-pred"><div class="lab">{escape(title)}</div>'
                f'<div class="val">{escape(label)}</div>{pct}{bar}'
                f'{pill(tier.upper() + " CONFIDENCE" if not patient else "AI certainty: " + tier, col)}'
                f'<div class="ms-mut" style="margin-top:8px">{escape(subtitle)}</div></div>', unsafe_allow_html=True)


def card_open(st, title):
    st.markdown(f'<div class="ms-card"><h4>{escape(title)}</h4>', unsafe_allow_html=True)


def card_close(st):
    st.markdown("</div>", unsafe_allow_html=True)


def feature_strip(st):
    st.markdown('<div class="ms-feat">'
                '<div><b>Deep learning</b><span>DenseNet121 + EfficientNet-B0 ensemble with one head per task.</span></div>'
                '<div><b>Explainable AI</b><span>Grad-CAM shows the regions that influenced each prediction.</span></div>'
                '<div><b>Interactive insights</b><span>Multi-slice analysis, MPR + 3D viewer, scan comparison.</span></div>'
                '<div><b>Privacy first</b><span>Patient IDs are stored hashed; images are not saved.</span></div></div>'
                '<div class="ms-foot">Built with Python · PyTorch · Streamlit · Plotly &nbsp;|&nbsp; '
                'For education and research. Not a medical device.</div>', unsafe_allow_html=True)

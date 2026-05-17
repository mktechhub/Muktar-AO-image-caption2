import os
import pickle
import warnings
import numpy as np
import streamlit as st
import gdown

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Afaan Oromo Image Captioner",
    page_icon="🖼️",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Syne:wght@700;800&display=swap');

:root {
    --violet:   #7c3aed;
    --pink:     #ec4899;
    --cyan:     #06b6d4;
    --lime:     #84cc16;
    --orange:   #f97316;
    --yellow:   #facc15;
    --bg:       #0a0914;
    --card:     #13111f;
    --card2:    #1a1730;
    --border:   rgba(255,255,255,0.07);
    --text:     #ede9fe;
    --muted:    #7a6fa8;
    --grad-hero:    linear-gradient(135deg, #7c3aed 0%, #ec4899 50%, #f97316 100%);
    --grad-caption: linear-gradient(135deg, #06b6d4 0%, #7c3aed 100%);
    --frame-c1: #7c3aed;
    --frame-c2: #ec4899;
    --frame-c3: #f97316;
    --frame-c4: #06b6d4;
    --frame-c5: #84cc16;
}

/* ── Reset ── */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Remove default padding so frame reaches edges ── */
.main > div { padding: 0 !important; }
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ════════════════════════════════════════════
   OUTER FRAME — fixed neon border on all 4 sides
   ════════════════════════════════════════════ */
.page-frame {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9999;
}

/* ── 4 glowing edge bars ── */
.frame-top, .frame-bottom, .frame-left, .frame-right {
    position: absolute;
    background: linear-gradient(90deg,
        var(--frame-c1), var(--frame-c2), var(--frame-c3),
        var(--frame-c4), var(--frame-c5), var(--frame-c1));
    background-size: 300% 100%;
    animation: frame-shift 6s linear infinite;
}
.frame-top    { top: 0;    left: 0; right: 0; height: 4px; }
.frame-bottom { bottom: 0; left: 0; right: 0; height: 4px; }
.frame-left   { top: 0;    left: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg,
        var(--frame-c1), var(--frame-c2), var(--frame-c3),
        var(--frame-c4), var(--frame-c5), var(--frame-c1));
    background-size: 100% 300%;
    animation: frame-shift-v 6s linear infinite;
}
.frame-right  { top: 0; right: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg,
        var(--frame-c5), var(--frame-c4), var(--frame-c3),
        var(--frame-c2), var(--frame-c1), var(--frame-c5));
    background-size: 100% 300%;
    animation: frame-shift-v 6s linear infinite reverse;
}
@keyframes frame-shift   { 0%{background-position:0% 0%} 100%{background-position:300% 0%} }
@keyframes frame-shift-v { 0%{background-position:0% 0%} 100%{background-position:0% 300%} }

/* ── Corner jewels ── */
.corner {
    position: absolute;
    width: 20px; height: 20px;
}
.corner::before, .corner::after {
    content: "";
    position: absolute;
    background: white;
    border-radius: 50%;
    animation: jewel-pulse 3s ease-in-out infinite;
}
.corner::before { width: 10px; height: 10px; top: 50%; left: 50%;
    transform: translate(-50%,-50%);
    box-shadow: 0 0 12px 4px rgba(255,255,255,0.7);
}
.corner::after  { width: 4px;  height: 4px;  top: 50%; left: 50%;
    transform: translate(-50%,-50%);
    background: white;
}
@keyframes jewel-pulse {
    0%,100% { box-shadow: 0 0 12px 4px rgba(255,255,255,0.6); }
    50%      { box-shadow: 0 0 22px 8px rgba(255,255,255,0.9); }
}
.corner-tl { top: -8px;  left: -8px;  }
.corner-tr { top: -8px;  right: -8px; }
.corner-bl { bottom: -8px; left: -8px;  }
.corner-br { bottom: -8px; right: -8px; }

/* ── Glow halos behind the 4 bars ── */
.frame-glow-top, .frame-glow-bottom {
    position: absolute; left: 0; right: 0; height: 60px; pointer-events: none;
}
.frame-glow-top    { top: 0;
    background: linear-gradient(180deg, rgba(124,58,237,0.18) 0%, transparent 100%); }
.frame-glow-bottom { bottom: 0;
    background: linear-gradient(0deg, rgba(6,182,212,0.15) 0%, transparent 100%); }
.frame-glow-left, .frame-glow-right {
    position: absolute; top: 0; bottom: 0; width: 60px; pointer-events: none;
}
.frame-glow-left  { left: 0;
    background: linear-gradient(90deg, rgba(236,72,153,0.13) 0%, transparent 100%); }
.frame-glow-right { right: 0;
    background: linear-gradient(270deg, rgba(132,204,22,0.10) 0%, transparent 100%); }

/* ── Inner decorative double-rule ── */
.frame-inner-top, .frame-inner-bottom {
    position: absolute; left: 24px; right: 24px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}
.frame-inner-top    { top: 14px;    }
.frame-inner-bottom { bottom: 14px; }
.frame-inner-left, .frame-inner-right {
    position: absolute; top: 24px; bottom: 24px; width: 1px;
    background: linear-gradient(180deg, transparent, rgba(255,255,255,0.12), transparent);
}
.frame-inner-left  { left: 14px;  }
.frame-inner-right { right: 14px; }

/* ── Dot-grid background ── */
.dot-grid-bg {
    position: fixed; inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.032) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none; z-index: -1;
}

/* ════════════════════════════════════════════
   INNER CONTENT WRAPPER
   ════════════════════════════════════════════ */
.content-wrap {
    max-width: 760px;
    margin: 0 auto;
    padding: 20px 32px 48px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.6rem 1rem 1.4rem;
    position: relative;
}
.hero-blob {
    position: absolute; top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 300px;
    background: radial-gradient(ellipse at 30% 50%,
        rgba(124,58,237,0.35) 0%, rgba(236,72,153,0.2) 40%, transparent 70%);
    filter: blur(40px); pointer-events: none;
}
.hero-blob2 {
    position: absolute; top: -20px; left: 62%;
    width: 280px; height: 200px;
    background: radial-gradient(ellipse, rgba(6,182,212,0.22) 0%, transparent 70%);
    filter: blur(35px); pointer-events: none;
}
.hero-icon-wrap {
    display: inline-flex; align-items: center; justify-content: center;
    width: 80px; height: 80px; border-radius: 24px;
    background: var(--grad-hero); font-size: 2.4rem; margin-bottom: 1.1rem;
    box-shadow: 0 0 40px rgba(124,58,237,0.55), 0 0 80px rgba(236,72,153,0.2);
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0px); }
    50%      { transform: translateY(-8px); }
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.8rem; font-weight: 800;
    background: var(--grad-hero);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.8rem; line-height: 1.12;
}
.hero-sub {
    font-size: 1rem; color: var(--muted); max-width: 460px;
    margin: 0 auto 1.8rem; line-height: 1.75;
}
.hero-sub b { color: #67e8f9; }

/* ── Badges ── */
.badges { display:flex; justify-content:center; gap:0.5rem; flex-wrap:wrap; margin-bottom:2rem; }
.badge  { padding:.3rem .9rem; border-radius:50px; font-size:.75rem; font-weight:800;
          letter-spacing:.04em; border:1px solid transparent; }
.b1 { background:rgba(124,58,237,.15); border-color:rgba(124,58,237,.45); color:#c4b5fd; }
.b2 { background:rgba(6,182,212,.12);  border-color:rgba(6,182,212,.40);  color:#67e8f9; }
.b3 { background:rgba(236,72,153,.12); border-color:rgba(236,72,153,.40); color:#f9a8d4; }
.b4 { background:rgba(132,204,22,.12); border-color:rgba(132,204,22,.40); color:#bef264; }

/* ── Rainbow bar ── */
.rainbow-bar {
    height: 2px;
    background: linear-gradient(90deg,
        #7c3aed,#ec4899,#f97316,#facc15,#84cc16,#06b6d4,#7c3aed);
    background-size: 200% 100%;
    border-radius: 2px; margin: 1.4rem 0 2rem;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer { 0%{background-position:0% 0%} 100%{background-position:200% 0%} }

/* ── Content section frame (inner card with ornate border) ── */
.section-frame {
    background: var(--card);
    border-radius: 24px;
    padding: 2rem;
    position: relative;
    margin-bottom: 1.5rem;

    /* Triple-layered border trick via box-shadow */
    box-shadow:
        0 0 0 1px rgba(124,58,237,0.35),
        0 0 0 3px rgba(236,72,153,0.12),
        0 0 0 5px rgba(6,182,212,0.06),
        0 12px 40px rgba(0,0,0,0.5),
        0 0 60px rgba(124,58,237,0.08);
}
/* Corner accent marks */
.section-frame::before {
    content: "";
    position: absolute; inset: 6px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.04);
    pointer-events: none;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--card2) !important;
    border: 2px dashed rgba(124,58,237,0.4) !important;
    border-radius: 18px !important;
    padding: 1.2rem !important;
    transition: all 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(236,72,153,0.6) !important;
    background: rgba(124,58,237,0.07) !important;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; }

/* ── Uploaded image ── */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 2px solid rgba(124,58,237,0.4) !important;
    box-shadow:
        0 0 0 4px rgba(124,58,237,0.10),
        0 0 30px rgba(124,58,237,0.25),
        0 0 60px rgba(236,72,153,0.10) !important;
    transition: all 0.4s;
}
[data-testid="stImage"] img:hover {
    box-shadow:
        0 0 0 4px rgba(124,58,237,0.18),
        0 0 50px rgba(124,58,237,0.45),
        0 0 90px rgba(6,182,212,0.15) !important;
    transform: scale(1.015);
}

/* ── Caption card ── */
.caption-card {
    background: var(--card2);
    border-radius: 18px;
    border: 1px solid rgba(124,58,237,0.28);
    padding: 1.6rem;
    margin-top: 0.4rem;
    position: relative; overflow: hidden;
    box-shadow: 0 0 0 2px rgba(236,72,153,0.06), 0 8px 24px rgba(0,0,0,0.4);
}
.caption-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--grad-caption);
}
.caption-card::after {
    content: ""; position: absolute;
    bottom: -55px; right: -55px; width: 170px; height: 170px;
    background: radial-gradient(circle, rgba(124,58,237,0.14) 0%, transparent 70%);
    pointer-events: none;
}
.cap-label {
    font-size: .68rem; font-weight: 800; letter-spacing: .18em;
    text-transform: uppercase;
    background: var(--grad-caption);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: .8rem;
}
.cap-text {
    font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700;
    color: var(--text); line-height: 1.6; margin-bottom: 1.1rem;
}
.chip-row { display:flex; gap:.45rem; flex-wrap:wrap; }
.chip {
    background: rgba(255,255,255,0.05); border: 1px solid var(--border);
    border-radius: 50px; padding: .2rem .65rem; font-size: .7rem; color: var(--muted);
}

/* ── Streamlit overrides ── */
[data-testid="stAlert"] { border-radius: 14px !important; }
.stSpinner > div { border-top-color: var(--violet) !important; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; color:var(--text) !important; }
hr  { border-color: var(--border) !important; }
footer { display: none !important; }

/* ── Footer ── */
.custom-footer {
    text-align: center; padding: 2rem 0 1rem; margin-top: 2.5rem;
    border-top: 1px solid var(--border);
}
.custom-footer p { font-size: .77rem; color: var(--muted); margin: .2rem 0; }
.custom-footer strong { color: #67e8f9; }
</style>

<!-- Dot grid bg -->
<div class="dot-grid-bg"></div>

<!-- ════ FULL-PAGE FRAME ════ -->
<div class="page-frame">
  <!-- Animated edge bars -->
  <div class="frame-top"></div>
  <div class="frame-bottom"></div>
  <div class="frame-left"></div>
  <div class="frame-right"></div>

  <!-- Glow halos -->
  <div class="frame-glow-top"></div>
  <div class="frame-glow-bottom"></div>
  <div class="frame-glow-left"></div>
  <div class="frame-glow-right"></div>

  <!-- Inner double rules -->
  <div class="frame-inner-top"></div>
  <div class="frame-inner-bottom"></div>
  <div class="frame-inner-left"></div>
  <div class="frame-inner-right"></div>

  <!-- Corner jewels -->
  <div class="corner corner-tl"></div>
  <div class="corner corner-tr"></div>
  <div class="corner corner-bl"></div>
  <div class="corner corner-br"></div>
</div>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE      = 224
FEATURE_DIM   = 49
FEATURE_DEPTH = 1280
DIGRAPHS      = ["dh", "ch", "sh", "ny", "ph", "ts"]

DRIVE_FILES = {
    "caption_model.keras"    : "1_ahJ11wNBCz5DM83XyyjMlbkefZoddt4",
    "feature_extractor.keras": "1FkxKKmduQFosbmkHVKzHf1zrXBb2mtNj",
    "tokenizer.pkl"          : "1oOPJLqsTGJmpqi1tVnZTPceJNGZcFlRX",
}


@st.cache_resource(show_spinner=False)
def download_models():
    for filename, file_id in DRIVE_FILES.items():
        if not os.path.exists(filename):
            with st.spinner(f"⬇️ Downloading {filename} …"):
                try:
                    gdown.download(
                        f"https://drive.google.com/uc?id={file_id}&confirm=t",
                        filename, quiet=False
                    )
                except Exception as e:
                    st.error(f"❌ Failed to download {filename}: {e}")
                    st.stop()


@st.cache_resource(show_spinner=False)
def load_models():
    download_models()
    import tensorflow as tf
    from keras.models import load_model
    from keras import ops as K
    from keras.layers import Dense, Layer

    class BahdanauAttention(Layer):
        def __init__(self, attn_dim, **kwargs):
            super().__init__(**kwargs)
            self.attn_dim = attn_dim
            self.W_feat   = Dense(attn_dim, use_bias=False, name="Wf")
            self.W_hid    = Dense(attn_dim, use_bias=False, name="Wh")
            self.V        = Dense(1,        use_bias=False, name="V")

        def call(self, features, hidden):
            h_exp   = K.expand_dims(hidden, axis=1)
            score   = self.V(K.tanh(self.W_feat(features) + self.W_hid(h_exp)))
            alpha   = K.softmax(score, axis=1)
            context = K.sum(alpha * features, axis=1)
            return context, K.squeeze(alpha, axis=-1)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"attn_dim": self.attn_dim})
            return cfg

    caption_model     = load_model(
        "caption_model.keras",
        custom_objects={"BahdanauAttention": BahdanauAttention},
        compile=False,
    )
    feature_extractor = load_model("feature_extractor.keras", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer, 40


def protect_digraphs(text):
    ph_map = {}
    for dg in DIGRAPHS:
        ph = f"XDIGX{dg.upper()}X"
        ph_map[ph] = dg
        text = text.replace(dg, ph)
    return text, ph_map

def restore_digraphs(text, ph_map):
    for ph, dg in ph_map.items():
        text = text.replace(ph, dg)
    return text


def generate_caption(image_path, caption_model, feature_extractor, tokenizer, max_length):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img_pp = preprocess_input(img_to_array(load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))))
    img_pp = np.expand_dims(img_pp, axis=0)
    feat   = feature_extractor.predict(img_pp, verbose=0).reshape(1, FEATURE_DIM, FEATURE_DEPTH)

    caption = "startseq"
    for _ in range(max_length):
        seq  = pad_sequences([tokenizer.texts_to_sequences([caption])[0]], maxlen=max_length)
        idx  = int(np.argmax(caption_model.predict([feat, seq], verbose=0)[0]))
        word = tokenizer.index_word.get(idx)
        if not word or word == "endseq":
            break
        caption += " " + word

    caption = caption.replace("startseq", "").strip()
    _, ph_map = protect_digraphs(caption)
    return restore_digraphs(caption, ph_map)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-blob"></div>
  <div class="hero-blob2"></div>
  <div class="hero-icon-wrap">🖼️</div>
  <h1 class="hero-title">Afaan Oromo<br>Image Captioner</h1>
  <p class="hero-sub">
    Upload any image and get an AI-generated caption in
    <b>Afaan Oromo</b> — Ethiopia's most widely spoken language.
  </p>
</div>

<div class="badges">
  <span class="badge b1">⚡ EfficientNetB0</span>
  <span class="badge b2">🔭 Bahdanau Attention</span>
  <span class="badge b3">🧬 LSTM Decoder</span>
  <span class="badge b4">🇪🇹 Afaan Oromo</span>
</div>

<div class="rainbow-bar"></div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("🚀 Initialising AI models … (first run may take a minute)"):
    caption_model, feature_extractor, tokenizer, max_length = load_models()

st.success("✅ Models ready — upload an image to generate a caption!")
st.markdown("<br>", unsafe_allow_html=True)

# ── Upload section in a decorative frame ──────────────────────────────────────
st.markdown('<div class="section-frame">', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "📂  Drop an image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG & PNG. The model generates an Afaan Oromo caption."
)

if uploaded is not None:
    tmp_path = "uploaded_image.jpg"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("✨ Generating your caption …"):
            try:
                caption = generate_caption(
                    tmp_path, caption_model, feature_extractor, tokenizer, max_length
                )
                st.markdown(f"""
<div class="caption-card">
  <div class="cap-label">✦ Generated Caption</div>
  <div class="cap-text">{caption}</div>
  <div class="chip-row">
    <span class="chip">📐 EfficientNetB0</span>
    <span class="chip">🔍 Attention</span>
    <span class="chip">🇪🇹 Afaan Oromo</span>
  </div>
</div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error generating caption: {e}")

st.markdown('</div>', unsafe_allow_html=True)  # close section-frame

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
  <p>Built with ❤️ using <strong>Streamlit</strong> · Afaan Oromo Image Captioning</p>
  <p>Design &amp; Developed by <strong>MUKTAR GETU</strong> &nbsp;·&nbsp; 📞 +251 923 009 973</p>
</div>
""", unsafe_allow_html=True)

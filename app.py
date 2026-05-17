import os
import re
import pickle
import warnings
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import gdown

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Afaan Oromo Image Captioner",
    page_icon="🖼️",
    layout="centered"
)

# ── Custom CSS — Luxury Dark Amber / Earth-tone aesthetic ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-deep:     #0d0a07;
    --bg-card:     #161109;
    --bg-glass:    rgba(255,210,120,0.04);
    --gold:        #e8b84b;
    --gold-light:  #f5d07a;
    --gold-dark:   #a07828;
    --amber:       #c47a1e;
    --cream:       #f0e6cc;
    --muted:       #7a6a50;
    --border:      rgba(232,184,75,0.18);
    --success-bg:  rgba(80,200,120,0.10);
    --success-txt: #6ddb9a;
    --shadow:      0 8px 40px rgba(0,0,0,0.6);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-deep) !important;
    color: var(--cream) !important;
}

/* ── Main container ── */
.main .block-container {
    max-width: 780px;
    padding: 2rem 2rem 4rem;
    background: var(--bg-deep);
}

/* ── Hero header ── */
.hero-wrap {
    text-align: center;
    padding: 3.5rem 0 2rem;
    position: relative;
}
.hero-wrap::before {
    content: "";
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(232,184,75,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon {
    font-size: 3.2rem;
    display: block;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 0 18px rgba(232,184,75,0.5));
    animation: pulse-glow 3s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { filter: drop-shadow(0 0 14px rgba(232,184,75,0.4)); }
    50%       { filter: drop-shadow(0 0 28px rgba(232,184,75,0.75)); }
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--gold-light), var(--amber));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.6rem;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: var(--muted);
    letter-spacing: 0.03em;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.7;
}
.hero-subtitle strong { color: var(--gold); font-weight: 500; }

/* ── Gold divider ── */
.gold-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2rem 0;
}
.gold-divider span {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold-dark), transparent);
}
.gold-divider em {
    color: var(--gold);
    font-style: normal;
    font-size: 1rem;
}

/* ── Model badge strip ── */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin: 1.4rem 0 2.2rem;
}
.badge {
    background: rgba(232,184,75,0.08);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 0.28rem 0.85rem;
    font-size: 0.78rem;
    color: var(--gold-light);
    letter-spacing: 0.04em;
    font-weight: 500;
}

/* ── Upload zone styling ── */
[data-testid="stFileUploader"] {
    background: var(--bg-glass) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-size: 0.88rem !important;
}

/* ── Image card ── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
    transition: transform 0.4s ease, box-shadow 0.4s ease;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.01);
    box-shadow: 0 12px 50px rgba(232,184,75,0.15) !important;
}

/* ── Caption result box ── */
.caption-card {
    background: linear-gradient(135deg, rgba(232,184,75,0.06), rgba(196,122,30,0.04));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.8rem;
    margin-top: 0.5rem;
    position: relative;
    overflow: hidden;
}
.caption-card::after {
    content: "✦";
    position: absolute;
    top: -18px; right: -12px;
    font-size: 5rem;
    color: rgba(232,184,75,0.05);
    pointer-events: none;
}
.caption-label {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--gold-dark);
    font-weight: 600;
    margin-bottom: 0.7rem;
}
.caption-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--cream);
    line-height: 1.55;
}
.meta-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}
.meta-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 50px;
    padding: 0.22rem 0.7rem;
    font-size: 0.72rem;
    color: var(--muted);
}

/* ── Success / error / spinner overrides ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border-left-width: 3px !important;
}
.stSpinner > div {
    border-top-color: var(--gold) !important;
}

/* ── Streamlit default element fixes ── */
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--cream) !important; }
hr { border-color: var(--border) !important; }
.stMarkdown p { color: var(--cream); }
footer { display: none !important; }

/* ── Footer ── */
.custom-footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
.custom-footer p {
    font-size: 0.75rem;
    color: var(--muted);
    margin: 0.15rem 0;
}
.custom-footer a { color: var(--gold-dark); text-decoration: none; }
.custom-footer strong { color: var(--gold); }
</style>
""", unsafe_allow_html=True)


# ── Constants (must match your training CFG) ──────────────────────────────────
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
            with st.spinner(f"Downloading {filename} …"):
                try:
                    url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
                    gdown.download(url, filename, quiet=False)
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
            alpha   = K.squeeze(alpha, axis=-1)
            return context, alpha

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

    max_length = 40
    return caption_model, feature_extractor, tokenizer, max_length


# ── Afaan Oromo digraph helpers ───────────────────────────────────────────────
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


# ── Caption generation ────────────────────────────────────────────────────────
def generate_caption(image_path, caption_model, feature_extractor,
                     tokenizer, max_length):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img_pil = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_pp  = preprocess_input(img_to_array(img_pil))
    img_pp  = np.expand_dims(img_pp, axis=0)

    feat = feature_extractor.predict(img_pp, verbose=0)
    feat = feat.reshape(1, FEATURE_DIM, FEATURE_DEPTH)

    caption = "startseq"
    for _ in range(max_length):
        seq  = tokenizer.texts_to_sequences([caption])[0]
        seq  = pad_sequences([seq], maxlen=max_length)
        pred = caption_model.predict([feat, seq], verbose=0)[0]
        idx  = int(np.argmax(pred))
        word = tokenizer.index_word.get(idx)
        if not word or word == "endseq":
            break
        caption += " " + word

    caption = caption.replace("startseq", "").strip()
    _, ph_map = protect_digraphs(caption)
    caption   = restore_digraphs(caption, ph_map)
    return caption


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <span class="hero-icon">🖼️</span>
  <h1 class="hero-title">Afaan Oromo<br>Image Captioner</h1>
  <p class="hero-subtitle">
    Upload any image and receive an AI-generated caption in
    <strong>Afaan Oromo</strong> — powered by a custom vision-language model.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Badge strip ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="badge-row">
  <span class="badge">⚡ EfficientNetB0</span>
  <span class="badge">🧠 Bahdanau Attention</span>
  <span class="badge">🔤 LSTM Decoder</span>
  <span class="badge">🇪🇹 Afaan Oromo</span>
</div>
""", unsafe_allow_html=True)

# ── Divider ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gold-divider">
  <span></span><em>✦</em><span></span>
</div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("Initialising models — please wait …"):
    caption_model, feature_extractor, tokenizer, max_length = load_models()

st.success("✅ Models ready — upload an image to begin.")

# ── File uploader ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG and PNG. The model will generate a caption in Afaan Oromo."
)

# ── Result ────────────────────────────────────────────────────────────────────
if uploaded is not None:
    tmp_path = "uploaded_image.jpg"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Generating caption …"):
            try:
                caption = generate_caption(
                    tmp_path, caption_model, feature_extractor,
                    tokenizer, max_length
                )
                st.markdown(f"""
<div class="caption-card">
  <div class="caption-label">Generated Caption</div>
  <div class="caption-text">{caption}</div>
  <div class="meta-row">
    <span class="meta-chip">📐 EfficientNetB0</span>
    <span class="meta-chip">🔍 Attention</span>
    <span class="meta-chip">🇪🇹 Afaan Oromo</span>
  </div>
</div>
""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating caption: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
  <p>Built with <strong>Streamlit</strong> · Afaan Oromo Image Captioning</p>
  <p>Design &amp; Developed by <strong>MUKTAR GETU</strong> &nbsp;·&nbsp; 📞 +251 923 009 973</p>
</div>
""", unsafe_allow_html=True)

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

# ── CSS — matches the HTML preview exactly ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700;800;900&family=Syne:wght@800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
    background: #0f0e17 !important;
    color: #ede9fe !important;
}

/* ── Remove Streamlit chrome padding ── */
.main > div { padding-top: 0 !important; }
.main .block-container {
    max-width: 720px !important;
    padding: 0 20px 60px !important;
    margin: 0 auto !important;
}

/* ══════════════════════════════════════
   WRAP  — the outer design container
   ══════════════════════════════════════ */
.wrap {
    background: #0f0e17;
    padding: 20px;
    border-radius: 16px;
    font-family: 'Nunito', sans-serif;
    max-width: 680px;
    margin: 0 auto;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 24px 0 16px;
    position: relative;
}
.blob1 {
    position: absolute; top: -20px; left: 50%;
    transform: translateX(-50%);
    width: 340px; height: 160px;
    background: radial-gradient(ellipse at 30% 50%,
        rgba(124,58,237,0.38) 0%, rgba(236,72,153,0.22) 45%, transparent 70%);
    filter: blur(28px); pointer-events: none;
}
.blob2 {
    position: absolute; top: 0; left: 62%;
    width: 200px; height: 130px;
    background: radial-gradient(ellipse, rgba(6,182,212,0.25) 0%, transparent 70%);
    filter: blur(24px); pointer-events: none;
}
.icon-box {
    display: inline-flex; align-items: center; justify-content: center;
    width: 62px; height: 62px; border-radius: 18px;
    background: linear-gradient(135deg, #7c3aed, #ec4899, #f97316);
    font-size: 1.8rem; margin-bottom: 12px;
    box-shadow: 0 0 30px rgba(124,58,237,0.55), 0 0 60px rgba(236,72,153,0.2);
    animation: float-icon 4s ease-in-out infinite;
}
@keyframes float-icon {
    0%,100% { transform: translateY(0px); }
    50%      { transform: translateY(-8px); }
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #7c3aed, #ec4899, #f97316);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 8px;
}
.hero-sub {
    font-size: 0.85rem; color: #8b7fb8; line-height: 1.7;
    max-width: 340px; margin: 0 auto 16px;
}
.hero-sub b { color: #67e8f9; }

/* ── Badges ── */
.badges {
    display: flex; justify-content: center; gap: 6px;
    flex-wrap: wrap; margin-bottom: 18px;
}
.badge {
    padding: 4px 12px; border-radius: 50px; font-size: 0.68rem;
    font-weight: 800; letter-spacing: 0.04em; border: 1px solid transparent;
}
.b1 { background: rgba(124,58,237,.18); border-color: rgba(124,58,237,.5);  color: #c4b5fd; }
.b2 { background: rgba(6,182,212,.14);  border-color: rgba(6,182,212,.45);  color: #67e8f9; }
.b3 { background: rgba(236,72,153,.14); border-color: rgba(236,72,153,.45); color: #f9a8d4; }
.b4 { background: rgba(132,204,22,.14); border-color: rgba(132,204,22,.45); color: #bef264; }

/* ── Rainbow bar ── */
.rainbow {
    height: 2px;
    background: linear-gradient(90deg,
        #7c3aed, #ec4899, #f97316, #facc15, #84cc16, #06b6d4, #7c3aed);
    background-size: 200% 100%; border-radius: 2px;
    margin: 0 0 18px; animation: shimmer 4s linear infinite;
}
@keyframes shimmer {
    0%   { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #1e1c2e !important;
    border: 2px dashed rgba(124,58,237,.45) !important;
    border-radius: 16px !important;
    padding: 18px !important;
    text-align: center !important;
    margin-bottom: 16px !important;
    transition: border-color .3s, background .3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(236,72,153,.6) !important;
    background: rgba(124,58,237,.06) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span { color: #8b7fb8 !important; font-size: 0.8rem !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: #8b7fb8 !important; }

/* ── Uploaded image ── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 2px solid rgba(124,58,237,.35) !important;
    box-shadow: 0 0 24px rgba(124,58,237,.22), 0 0 50px rgba(236,72,153,.1) !important;
    transition: transform .3s, box-shadow .3s;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.012);
    box-shadow: 0 0 40px rgba(124,58,237,.4), 0 0 70px rgba(236,72,153,.15) !important;
}

/* ── Caption card ── */
.cap-card {
    background: #1e1c2e;
    border-radius: 14px;
    border: 1px solid rgba(124,58,237,.28);
    padding: 16px;
    position: relative; overflow: hidden;
    height: 100%;
}
.cap-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #06b6d4, #7c3aed);
}
.cap-label {
    font-size: 0.62rem; font-weight: 800; letter-spacing: .18em;
    text-transform: uppercase;
    background: linear-gradient(90deg, #06b6d4, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 8px; display: block;
}
.cap-text {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem; font-weight: 700;
    color: #ede9fe; line-height: 1.55; margin-bottom: 10px;
}
.chips { display: flex; gap: 5px; flex-wrap: wrap; }
.chip {
    background: rgba(255,255,255,.05);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 50px; padding: 2px 8px;
    font-size: 0.62rem; color: #8b7fb8;
}

/* ── Streamlit alert / spinner overrides ── */
[data-testid="stAlert"]   { border-radius: 14px !important; }
.stSpinner > div          { border-top-color: #7c3aed !important; }
.stColumns [data-testid="column"] { gap: 14px; }

/* ── Hide default Streamlit footer ── */
footer { display: none !important; }

/* ── Custom footer ── */
.custom-footer {
    text-align: center;
    border-top: 1px solid rgba(255,255,255,.07);
    margin-top: 20px; padding-top: 14px;
}
.custom-footer p { font-size: 0.7rem; color: #8b7fb8; margin: 2px 0; }
.custom-footer strong { color: #67e8f9; }
</style>
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
#  UI  — mirrors the HTML preview layout 1:1
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero + badges + rainbow bar ───────────────────────────────────────────────
st.markdown("""
<div class="wrap">
  <div class="hero">
    <div class="blob1"></div>
    <div class="blob2"></div>
    <div class="icon-box">🖼️</div>
    <div class="hero-title">Afaan Oromo<br>Image Captioner</div>
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

  <div class="rainbow"></div>
</div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("🚀 Initialising AI models … (first run may take a minute)"):
    caption_model, feature_extractor, tokenizer, max_length = load_models()

st.success("✅ Models ready — upload an image to generate a caption!")
st.markdown("<br>", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📂  Drop an image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG & PNG. The model generates an Afaan Oromo caption."
)

# ── Result ────────────────────────────────────────────────────────────────────
if uploaded is not None:
    tmp_path = "uploaded_image.jpg"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("✨ Generating caption …"):
            try:
                caption = generate_caption(
                    tmp_path, caption_model, feature_extractor, tokenizer, max_length
                )
                st.markdown(f"""
<div class="cap-card">
  <span class="cap-label">✦ Generated Caption</span>
  <div class="cap-text">{caption}</div>
  <div class="chips">
    <span class="chip">📐 EfficientNetB0</span>
    <span class="chip">🔍 Attention</span>
    <span class="chip">🇪🇹 Afaan Oromo</span>
  </div>
</div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error generating caption: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
  <p>Built with ❤️ using <strong>Streamlit</strong> · Afaan Oromo Image Captioning</p>
  <p>Design &amp; Developed by <strong>MUKTAR GETU</strong> &nbsp;·&nbsp; 📞 +251 923 009 973</p>
</div>
""", unsafe_allow_html=True)

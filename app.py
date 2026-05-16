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

# ── Constants (must match your training CFG) ──────────────────────────────────
IMG_SIZE      = 224
FEATURE_DIM   = 49       # 7×7
FEATURE_DEPTH = 1280     # EfficientNetB0 channels
DIGRAPHS      = ["dh", "ch", "sh", "ny", "ph", "ts"]

# ── Google Drive file IDs — REPLACE THESE WITH YOUR ACTUAL IDs ───────────────
DRIVE_FILES = {
    "caption_model.keras"    : "1_ahJ11wNBCz5DM83XyyjMlbkefZoddt4",
    "feature_extractor.keras": "1FkxKKmduQFosbmkHVKzHf1zrXBb2mtNj",
    "tokenizer.pkl"          : "1oOPJLqsTGJmpqi1tVnZTPceJNGZcFlRX",
}

# ── Download models from Google Drive ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_models():
    for filename, file_id in DRIVE_FILES.items():
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {filename} …"):
                try:
                    url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
                    gdown.download(url, filename, quiet=False, fuzzy=True)
                except Exception as e:
                    st.error(f"❌ Failed to download {filename}: {e}")
                    st.stop()

# ── Load models (cached so they load only once) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    download_models()

    # Import heavy deps inside cached function
    import tensorflow as tf
    from keras.models import load_model
    from keras import ops as K
    from keras.layers import Dense, Layer

    # ── Re-define BahdanauAttention (needed as custom_object) ─────────────────
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

    caption_model = load_model(
        "caption_model.keras",
        custom_objects={"BahdanauAttention": BahdanauAttention},
        compile=False,
    )
    feature_extractor = load_model("feature_extractor.keras", compile=False)

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    max_length = 42  # same value used during training

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

    # Load & preprocess
    img_pil = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_pp  = preprocess_input(img_to_array(img_pil))
    img_pp  = np.expand_dims(img_pp, axis=0)

    # Extract spatial features (1, 49, 1280)
    feat = feature_extractor.predict(img_pp, verbose=0)
    feat = feat.reshape(1, FEATURE_DIM, FEATURE_DEPTH)

    # Greedy decode
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

    # Restore Afaan Oromo digraphs if needed
    _, ph_map = protect_digraphs(caption)
    caption   = restore_digraphs(caption, ph_map)

    return caption


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("🖼️ Afaan Oromo Image Captioner")
st.markdown(
    "Upload an image and the model will generate a caption in **Afaan Oromo** "
    "using **EfficientNetB0 + Bahdanau Attention + LSTM**."
)
st.divider()

# Load models on first run
with st.spinner("Loading models … (this may take a minute on first run)"):
    caption_model, feature_extractor, tokenizer, max_length = load_models()

st.success("✅ Models loaded!")

# File uploader
uploaded = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"],
    help="Upload any image — the model will generate an Afaan Oromo caption."
)

if uploaded is not None:
    # Save temp file
    tmp_path = "uploaded_image.jpg"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Generating caption …"):
            try:
                caption = generate_caption(
                    tmp_path, caption_model, feature_extractor,
                    tokenizer, max_length
                )
                st.markdown("### 📝 Generated Caption")
                st.success(caption)
                st.markdown("---")
                st.markdown("**Model:** EfficientNetB0 + Bahdanau Attention + LSTM")
                st.markdown("**Language:** Afaan Oromo")
            except Exception as e:
                st.error(f"Error generating caption: {e}")

st.divider()
st.caption("Built with Streamlit · Afaan Oromo Image Captioning Project")

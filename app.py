import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐾", layout="centered")
st.title("🐾 Cats vs Dogs Classifier")
st.caption("MobileNetV2 Deep Transfer Learning · AVIP 2026")

IMG_SIZE = (160, 160)
MODEL_PATH = "models/cats_dogs_mobilenetv2.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    model_ready = model is not None
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_ready = False

if not model_ready:
    st.warning("Model file not found at models/cats_dogs_mobilenetv2.keras. Please ensure the model is present.")

uploaded = st.file_uploader("Upload a cat or dog photograph", type=["jpg", "jpeg", "png", "webp"])

if uploaded and model_ready:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    resized_img = img.resize(IMG_SIZE)
    arr = np.array(resized_img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)

    logit = model.predict(arr, verbose=0)[0][0]
    p_dog = float(tf.sigmoid(logit).numpy())
    p_cat = 1.0 - p_dog
    label = "DOG 🐕" if p_dog >= 0.50 else "CAT 🐈"

    st.markdown(f"### Prediction: **{label}**")
    st.progress(p_dog)

    col1, col2 = st.columns(2)
    col1.metric("Cat Probability P(Cat)", f"{p_cat * 100:.2f}%")
    col2.metric("Dog Probability P(Dog)", f"{p_dog * 100:.2f}%")

st.markdown("---")
st.subheader("Try a Real Held-Out Test Photo")
sample_files = {
    "Cat Sample 1 (cat_01.jpg)": "outputs/sample_inferences/cat_01.jpg",
    "Cat Sample 2 (cat_02.jpg)": "outputs/sample_inferences/cat_02.jpg",
    "Dog Sample 1 (dog_01.jpg)": "outputs/sample_inferences/dog_01.jpg",
    "Dog Sample 2 (dog_02.jpg)": "outputs/sample_inferences/dog_02.jpg"
}

chosen_sample = st.selectbox("Select a verified test photo:", ["Select a sample..."] + list(sample_files.keys()))
if chosen_sample != "Select a sample..." and model_ready:
    sample_path = sample_files[chosen_sample]
    if os.path.exists(sample_path):
        s_img = Image.open(sample_path).convert("RGB")
        st.image(s_img, caption=f"Test Image: {chosen_sample}", width=320)
        
        resized_img = s_img.resize(IMG_SIZE)
        arr = np.array(resized_img, dtype=np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        arr = np.expand_dims(arr, 0)
        
        logit = model.predict(arr, verbose=0)[0][0]
        p_dog = float(tf.sigmoid(logit).numpy())
        p_cat = 1.0 - p_dog
        pred_label = "DOG 🐕" if p_dog >= 0.50 else "CAT 🐈"
        conf = p_dog if "DOG" in pred_label else p_cat
        st.markdown(f"**Predicted Class:** `{pred_label}` (Confidence: {conf*100:.2f}%)")


import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = load_model("face_mask_model.keras")

st.title("😷 Face Mask Detector")
st.write("Upload an image to check if the person is wearing a mask or not.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    with st.container():
        st.subheader("📷 Uploaded Image")
        st.image(img, use_container_width=True)

    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    if confidence < 0.5:
        result = "😷 Wearing Mask"
        confidence = 1 - confidence
        st.success(f"✅ Prediction: {result}")
    else:
        result = "🙁 No Mask"
        st.error(f"Prediction: {result}")

    st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")


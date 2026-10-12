
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="Smart Waste Vision",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Smart Waste Vision")
st.write("Upload an image to analyze it using computer vision and AI.")

@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="microsoft/resnet-50"
    )

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    st.subheader("1. Original Image")
    st.image(image, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2. Grayscale Image")
        st.image(gray, use_container_width=True)

    with col2:
        st.subheader("3. Edge Detection")
        st.image(edges, use_container_width=True)

    st.subheader("4. AI Image Classification")

    if st.button("Analyze Image", type="primary"):
        try:
            with st.spinner("Loading AI model and analyzing image..."):
                model = load_model()
                predictions = model(image, top_k=3)

            st.success("Image analysis completed!")

            for item in predictions:
                st.write(
                    f"**{item['label']}** — "
                    f"{item['score'] * 100:.2f}% confidence"
                )
                st.progress(float(item["score"]))

            st.warning(
                "This is a general object-recognition model, "
                "not a waste-specific classifier. Its predictions "
                "may not identify the material correctly."
            )

        except Exception as error:
            st.error(f"Could not analyze the image: {error}")
            st.info(
                "Check your internet connection and try again. "
                "The model downloads the first time it is used."
            )

else:
    st.info("Upload an image to get started.")

st.divider()
st.caption("Smart Waste Vision | Computer Vision Project")

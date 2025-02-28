import streamlit as st
import requests

st.markdown("<h1 style='text-align: center; font-size: 42px;'>AI Image Classifier</h1>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)

    if response.status_code == 200:
        predictions = response.json()["predictions"]
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.subheader("Predictions:")
        for pred in predictions:
            st.markdown(f"<h3 style='font-size: 24px;'>{pred['label']} - {pred['probability']*100:.2f}%</h3>", unsafe_allow_html=True)

    else:
        st.error("Error processing the image.")

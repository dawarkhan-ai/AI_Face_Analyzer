import streamlit as st
import cv2
import numpy as np
from landmarks import get_landmarks, draw_landmarks
from analysis import calculate_face_metrics
from analysis import calculate_symmetry
from analysis import classify_face_shape
from analysis import golden_ratio_analysis

st.title("AI Face Geometry Analyzer")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    landmarks = get_landmarks(image)

    if landmarks:
        # show facial points
        image_with_landmarks = draw_landmarks(image.copy(), landmarks)
        st.image(image_with_landmarks, channels="BGR")

        # show facial measurements
        metrics = calculate_face_metrics(landmarks, image.shape)

        st.write("### Face Measurements 📏")
        for key, value in metrics.items():
            st.write(f"{key}: {value:.2f}")

        # show symmetry score
        symmetry_score = calculate_symmetry(landmarks, image.shape)

        st.write("### Face Symmetry 🪞")
        st.write(f"Symmetry Score: {symmetry_score:.2f} / 100")

        # show face shape
        face_shape = classify_face_shape(landmarks, image.shape)

        st.write("### Face Shape 🧠")
        st.write(f"Detected Shape: {face_shape}")

        # show golden ratio
        ratio, score = golden_ratio_analysis(landmarks, image.shape)

        st.write("### Golden Ratio Analysis ✨")
        st.write(f"Ratio: {ratio:.2f}")
        st.write(f"Golden Score: {score:.2f} / 100")

    else:
        st.write("Face not detected ❌")
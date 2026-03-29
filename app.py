import streamlit as st
import cv2
import numpy as np

from landmarks import get_landmarks, draw_landmarks
from analysis import (
    calculate_face_metrics,
    calculate_symmetry,
    classify_face_shape,
    golden_ratio_analysis
)

st.title("AI Face Geometry Analyzer")

# -------------------------------
# Mode Selection
# -------------------------------
mode = st.radio("Select Mode", ["Image Upload 📸", "Real-Time Camera 🎥"])


# =========================================================
# 📸 IMAGE MODE
# =========================================================
if mode == "Image Upload 📸":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        landmarks = get_landmarks(image)

        if landmarks:
            # Landmarks draw
            image_with_landmarks = draw_landmarks(image.copy(), landmarks)
            st.image(image_with_landmarks, channels="BGR")

            # Measurements
            metrics = calculate_face_metrics(landmarks, image.shape)

            st.write("### Face Measurements 📏")
            if metrics:
                for key, value in metrics.items():
                    st.write(f"{key}: {value:.2f}")

            # Symmetry
            symmetry_score = calculate_symmetry(landmarks, image.shape)
            st.write("### Face Symmetry 🪞")
            if symmetry_score is not None:
                st.write(f"Symmetry Score: {symmetry_score:.2f} / 100")

            # Face Shape
            face_shape = classify_face_shape(landmarks, image.shape)
            st.write("### Face Shape 🧠")
            st.write(f"Detected Shape: {face_shape}")

            # Golden Ratio
            ratio, score = golden_ratio_analysis(landmarks, image.shape)
            st.write("### Golden Ratio Analysis ✨")
            if ratio is not None:
                st.write(f"Ratio: {ratio:.2f}")
                st.write(f"Golden Score: {score:.2f} / 100")

        else:
            st.write("Face not detected ❌")


# =========================================================
# 🎥 REAL-TIME MODE
# =========================================================
elif mode == "Real-Time Camera 🎥":

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])
    metrics_text = st.empty()

    camera = cv2.VideoCapture(0)

    # -------------------------
    # Smoothing variables
    # -------------------------
    alpha = 0.3  # smoothing factor

    prev_width = None
    prev_height = None
    prev_eye = None
    prev_symmetry = None
    prev_ratio = None

    threshold = 2  # ignore very small changes

    while run:
        ret, frame = camera.read()

        if not ret:
            st.write("Camera not working ❌")
            break

        landmarks = get_landmarks(frame)

        if landmarks:
            frame = draw_landmarks(frame, landmarks)

            # Raw values
            metrics = calculate_face_metrics(landmarks, frame.shape)
            symmetry = calculate_symmetry(landmarks, frame.shape)
            face_shape = classify_face_shape(landmarks, frame.shape)
            ratio, score = golden_ratio_analysis(landmarks, frame.shape)

            width = metrics["Face Width"]
            height = metrics["Face Height"]
            eye = metrics["Eye Distance"]

            # -------------------------
            # Exponential smoothing
            # -------------------------
            def smooth(current, prev):
                if prev is None:
                    return current
                if abs(current - prev) < threshold:
                    return prev
                return alpha * current + (1 - alpha) * prev

            width = smooth(width, prev_width)
            height = smooth(height, prev_height)
            eye = smooth(eye, prev_eye)
            symmetry = smooth(symmetry, prev_symmetry)
            ratio = smooth(ratio, prev_ratio)

            # Update previous values
            prev_width = width
            prev_height = height
            prev_eye = eye
            prev_symmetry = symmetry
            prev_ratio = ratio

            # -------------------------
            # Display
            # -------------------------
            metrics_text.markdown(f"""
            ### 📊 Live Analysis (Smoothed)

            **Face Width:** {width:.2f}  
            **Face Height:** {height:.2f}  
            **Eye Distance:** {eye:.2f}  

            **Symmetry Score:** {symmetry:.2f} / 100  

            **Face Shape:** {face_shape}  

            **Golden Ratio:** {ratio:.2f}  
            **Golden Score:** {score:.2f} / 100  
            """)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()
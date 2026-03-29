# 🧠 AI Face Geometry Analyzer

An AI-based computer vision project that analyzes human facial structure using geometric and mathematical techniques.

This application detects facial landmarks and performs multiple analyses such as face measurements, symmetry evaluation, face shape classification, and golden ratio comparison.

---

## 🚀 Features

- 📸 Upload image for analysis
- 👤 Face detection using MediaPipe
- 📍 Facial landmark detection (468 points)
- 📏 Face geometry measurements
  - Face Width
  - Face Height
  - Eye Distance
- 🪞 Face symmetry analysis (Score out of 100)
- 🧠 Face shape classification
  - Round
  - Oval
  - Square
- ✨ Golden ratio analysis (1.618 comparison)
- 📊 Real-time results display using Streamlit

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – Image processing
- **MediaPipe (Tasks API)** – Face landmark detection
- **NumPy** – Mathematical calculations
- **Streamlit** – Interactive UI

---

## 📂 Project Structure
```
AI-Face-Analyzer/
│
├── app.py # Streamlit UI
├── landmarks.py # Face landmark detection
├── analysis.py # Geometry + symmetry + ratio logic
├── face_landmarker.task # MediaPipe model file
├── requirements.txt

```

---

## ⚙️ Installation

1. Clone the repository
```
git clone https://github.com/dawarkhan-ai/AI_Face_Analyzer

cd AI-Face-Analyzer
```

2. Create virtual environment
```
python -m venv venv
venv\Scripts\activate # Windows
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Download MediaPipe model

Download:
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

Place it in project root folder.

---

## ▶️ Run the App
`streamlit run app.py`


---

## 📊 How It Works

1. User uploads an image
2. Image is processed using OpenCV
3. MediaPipe detects facial landmarks
4. Geometry calculations are performed
5. Symmetry and ratios are analyzed
6. Results are displayed on screen

---

## 🧠 Concepts Used

- Computer Vision
- Facial Landmark Detection
- Euclidean Distance Calculation
- Symmetry Analysis
- Golden Ratio (1.618)
- Rule-based Classification

---

## ⚠️ Limitations

- Accuracy depends on image quality and angle
- Works best with front-facing images
- Face shape classification is rule-based (not ML-trained)

---

## 🚀 Future Improvements

- Real-time webcam analysis
- Deep learning-based face shape detection
- 3D face analysis
- Mobile app integration
- PDF report generation

---

## 👨‍💻 Author

**MD Dawar Khan**

---

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model path
MODEL_PATH = "face_landmarker.task"

# get facial landmarks/points
def get_landmarks(image):

    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Create options
    BaseOptions = python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE
    )

    # Create detector
    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            return result.face_landmarks[0]

    return None

# draw facial points 
def draw_landmarks(image, landmarks):
    h, w, _ = image.shape

    for point in landmarks:
        x = int(point.x * w)
        y = int(point.y * h)

        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    return image
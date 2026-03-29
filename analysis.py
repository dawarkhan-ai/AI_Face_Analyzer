import numpy as np

def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


# Landmarks → pixel convert
def get_pixel_coordinates(landmarks, image_shape):
    h, w, _ = image_shape

    coords = []
    for point in landmarks:
        x = int(point.x * w)
        y = int(point.y * h)
        coords.append((x, y))

    return coords

# Measurements Calculation
def calculate_face_metrics(landmarks, image_shape):
    coords = get_pixel_coordinates(landmarks, image_shape)

    # Important points
    left_cheek = coords[234]
    right_cheek = coords[454]
    forehead = coords[10]
    chin = coords[152]
    left_eye = coords[33]
    right_eye = coords[263]

    # Measurements
    face_width = calculate_distance(left_cheek, right_cheek)
    face_height = calculate_distance(forehead, chin)
    eye_distance = calculate_distance(left_eye, right_eye)

    return {
        "Face Width": face_width,
        "Face Height": face_height,
        "Eye Distance": eye_distance
    }

# Symmetry calculation
def calculate_symmetry(landmarks, image_shape):
    coords = get_pixel_coordinates(landmarks, image_shape)

    h, w, _ = image_shape
    mid_x = w // 2   # face ka vertical center

    differences = []

    for (x, y) in coords:
        # Mirror point ka x
        mirrored_x = 2 * mid_x - x

        # Difference
        diff = abs(x - mirrored_x)
        differences.append(diff)

    avg_diff = sum(differences) / len(differences)

    # Normalize score (approx)
    symmetry_score = max(0, 100 - (avg_diff / w) * 100)

    return symmetry_score

# face shape calculation
def classify_face_shape(landmarks, image_shape):
    coords = get_pixel_coordinates(landmarks, image_shape)

    # Important points
    left_cheek = coords[234]
    right_cheek = coords[454]
    forehead = coords[10]
    chin = coords[152]
    jaw_left = coords[172]
    jaw_right = coords[397]

    # Measurements
    face_width = calculate_distance(left_cheek, right_cheek)
    face_height = calculate_distance(forehead, chin)
    jaw_width = calculate_distance(jaw_left, jaw_right)

    ratio = face_height / face_width

    # Simple classification
    if 0.95 <= ratio <= 1.05:
        return "Round Face"
    elif ratio > 1.05:
        return "Oval Face"
    elif jaw_width > face_width * 0.75:
        return "Square Face"
    else:
        return "Undefined"
    
# Golden ratio calculation
def golden_ratio_analysis(landmarks, image_shape):
    coords = get_pixel_coordinates(landmarks, image_shape)

    # Points
    left_cheek = coords[234]
    right_cheek = coords[454]
    forehead = coords[10]
    chin = coords[152]

    # Measurements
    face_width = calculate_distance(left_cheek, right_cheek)
    face_height = calculate_distance(forehead, chin)

    ratio = face_height / face_width

    # Golden ratio comparison
    golden_ratio = 1.618
    difference = abs(ratio - golden_ratio)

    # Score (simple)
    score = max(0, 100 - difference * 100)

    return ratio, score
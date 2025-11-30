import numpy as np
import face_recognition
from PIL import Image

# small helpers
def euclidean_distance(p1, p2):
    # standard distance between two (x,y) points
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def polygon_area(points):
    # compute polygon area using the shoelace formula
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(points)-1)))

def face_lighting_level(image):
    # average brightness of the whole face area
    gray = np.mean(image, axis=2)
    return float(np.mean(gray))

# main feature extraction routine
def analyze_face(image_path, metadata=None):
    """
    Extracts geometric features (eye distance, lip area, lighting) for a face image.
    Can include optional metadata (like gender or id) in the return dictionary.
    """
    try:
        image = face_recognition.load_image_file(image_path)
        landmarks = face_recognition.face_landmarks(image)
    except Exception as e:
        print(f"Failed to analyze {image_path}: {e}")
        return None

    if not landmarks:
        print(f"No faces found in {image_path}")
        return None

    face = landmarks[0]  # assume one face per image
    left_eye_center = np.mean(face["left_eye"], axis=0)
    right_eye_center = np.mean(face["right_eye"], axis=0)

    eye_distance = euclidean_distance(left_eye_center, right_eye_center)
    lip_area = polygon_area(face["top_lip"]) + polygon_area(face["bottom_lip"])
    lighting_score = face_lighting_level(image)

    info = {
        "image_path": image_path,
        "eye_distance": eye_distance,
        "lip_area": lip_area,
        "lighting_score": lighting_score,
    }

    if metadata:
        info.update(metadata)

    return info

# shared geometry comparison for both tests
def compare_geometry(face_a, face_b):
    """
    Compare two faces using normalized geometric difference
    (eye distance and lip area). Returns a similarity score 0–1.
    """
    if not face_a or not face_b:
        return 0.0
    deye = abs(face_a["eye_distance"] - face_b["eye_distance"]) / (face_b["eye_distance"] + 1e-9)
    dlip = abs(face_a["lip_area"] - face_b["lip_area"]) / (face_b["lip_area"] + 1e-9)
    diff = np.sqrt(deye**2 + dlip**2)
    return float(1 / (1 + diff))  # 1 = identical, 0 = very different

# standalone use (for quick testing)
if __name__ == "__main__":
    data = analyze_face("./faces/01/image_0001.jpg", {"id": "01", "gender": "female"})
    if data:
        print("\nExtracted features:")
        for k, v in data.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, (int, float)) else f"  {k}: {v}")


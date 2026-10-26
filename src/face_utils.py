"""
Face Detection and Recognition Utilities (Final Stage 5 - Debugged & Optimized).

This module contains helper functions for:
- Initializing OpenCV Haar Cascade face detector.
- Loading the pre-trained deep learning face recognition model (SFace ONNX).
- Generating 128-dimensional L2-normalized deep face embeddings.
- Extracting and storing multi-sample prototype embeddings for all registered individuals.
- Comparing live query face embeddings with registered embeddings using cosine similarity.
- Identifying known persons vs. "Unknown Person" based on calibrated confidence thresholds.
- Detailed debugging metrics and diagnostics for face matching.
- Safely managing webcam video capture and drawing recognition badges.
"""

import os
import re
import sys
import pickle
import urllib.request
import cv2
import numpy as np

# Default file paths
SFACE_MODEL_PATH = "models/face_recognition_sface_2021dec.onnx"
SFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
EMBEDDINGS_CACHE_PATH = "models/embeddings.pkl"
DEFAULT_RECOGNITION_THRESHOLD = 0.50  # Calibrated for unaligned Haar crops + SFace
DEFAULT_AMBIGUITY_MARGIN = 0.08       # Minimum separation required between top 2 candidates


# ---------------------------------------------------------
# Stage 2: Face Detection & Camera Management
# ---------------------------------------------------------

def load_face_cascade():
    """
    Load the pre-trained OpenCV Haar Cascade frontal face detector.

    Returns:
        cv2.CascadeClassifier: Loaded cascade classifier object.
    Raises:
        RuntimeError: If the cascade classifier XML cannot be loaded.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError(
            f"Failed to load Haar Cascade face detector from: {cascade_path}"
        )
    return face_cascade


def detect_faces(frame: np.ndarray, face_cascade, scale_factor: float = 1.1, min_neighbors: int = 5):
    """
    Detect faces in a single video frame.

    Args:
        frame (np.ndarray): The BGR image frame from the webcam.
        face_cascade: The loaded face cascade detector.
        scale_factor (float): How much the image size is reduced at each image scale.
        min_neighbors (int): How many neighbors each candidate rectangle should retain.

    Returns:
        list of tuple: List of detected bounding boxes as (x, y, w, h).
    """
    if frame is None or face_cascade is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces


def draw_faces(frame: np.ndarray, faces, color: tuple = (0, 255, 0), thickness: int = 2, custom_label: str = None):
    """
    Draw bounding boxes and labels around detected faces on the frame.
    """
    annotated = frame.copy()

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        label = custom_label if custom_label else f"Face {i + 1}"
        label_pos = (x, max(y - 10, 20))
        cv2.putText(
            annotated,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    return annotated


def get_camera(camera_index: int = 0):
    """
    Attempt to initialize and open the webcam at the given device index.
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()

        # On macOS, try the AVFOUNDATION backend if default fails
        if sys.platform == "darwin":
            cap_mac = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
            if cap_mac is not None and cap_mac.isOpened():
                return cap_mac
            if cap_mac is not None:
                cap_mac.release()

        return None
    except Exception:
        return None


def release_camera(cap):
    """
    Safely release and close the OpenCV VideoCapture resource.
    """
    if cap is not None:
        try:
            if cap.isOpened():
                cap.release()
        except Exception:
            pass


# ---------------------------------------------------------
# Stage 3: Face Registration & Dataset Helper Functions
# ---------------------------------------------------------

def sanitize_name(name: str) -> str:
    """
    Sanitize person's name for safe folder creation.
    Replaces spaces with underscores and removes illegal filename characters.
    """
    clean = name.strip()
    clean = re.sub(r"[^\w\s-]", "", clean)
    clean = re.sub(r"[\s]+", "_", clean)
    return clean


def crop_face(frame: np.ndarray, face_box: tuple, target_size: tuple = (112, 112)) -> np.ndarray:
    """
    Crop face region of interest (ROI) from frame and resize it.
    """
    x, y, w, h = face_box
    h_frame, w_frame = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_frame, x + w)
    y2 = min(h_frame, y + h)

    face_roi = frame[y1:y2, x1:x2]
    resized_face = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
    return resized_face


def save_face_sample(face_crop: np.ndarray, person_name: str, sample_index: int, base_dir: str = "data/faces") -> str:
    """
    Save cropped face sample to the person's directory with high JPEG quality.
    """
    person_dir = os.path.join(base_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    filename = f"{person_name}_{sample_index:03d}.jpg"
    filepath = os.path.join(person_dir, filename)

    cv2.imwrite(filepath, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return filepath


def get_registered_users(base_dir: str = "data/faces") -> dict:
    """
    Scan dataset directory and return registered people with their sample counts.
    """
    if not os.path.exists(base_dir):
        return {}

    users = {}
    valid_exts = {".jpg", ".jpeg", ".png"}

    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name.lower()):
        if entry.is_dir():
            count = sum(
                1 for f in os.scandir(entry.path)
                if f.is_file() and os.path.splitext(f.name)[1].lower() in valid_exts
            )
            if count > 0:
                users[entry.name] = count

    return users


def delete_registered_user(person_name: str, base_dir: str = "data/faces") -> bool:
    """
    Delete a registered user's folder and remove cached embeddings.
    """
    person_dir = os.path.join(base_dir, person_name)
    if os.path.exists(person_dir) and os.path.isdir(person_dir):
        import shutil
        shutil.rmtree(person_dir)
        if os.path.exists(EMBEDDINGS_CACHE_PATH):
            try:
                os.remove(EMBEDDINGS_CACHE_PATH)
            except Exception:
                pass
        return True
    return False


# ---------------------------------------------------------
# Stage 4 & 5: Pretrained Deep Learning Face Recognition
# ---------------------------------------------------------

def ensure_model_exists(model_path: str = SFACE_MODEL_PATH) -> str:
    """
    Check if the pre-trained SFace ONNX deep learning model exists locally.
    If missing, downloads it from the official repository.
    """
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(SFACE_MODEL_URL, model_path)
        except Exception as e:
            raise RuntimeError(
                f"Pretrained model '{model_path}' not found and automatic download failed: {e}"
            )
    return model_path


def load_face_recognizer(model_path: str = SFACE_MODEL_PATH):
    """
    Load the pre-trained SFace deep learning model via OpenCV's DNN module.

    Returns:
        cv2.dnn.Net: Loaded deep neural network for feature extraction.
    """
    ensure_model_exists(model_path)
    net = cv2.dnn.readNetFromONNX(model_path)
    return net


def get_face_embedding(face_img: np.ndarray, recognizer_net) -> np.ndarray:
    """
    Generate a 128-dimensional L2-normalized feature embedding for a face image.

    How it works:
    1. Preprocesses the face image into a 112x112 RGB blob.
    2. Passes it through the deep convolutional neural network (SFace).
    3. Obtains a 128-dimensional feature vector and applies L2-normalization
       so that cosine similarity can be computed via standard dot product.

    Args:
        face_img (np.ndarray): Cropped BGR face image.
        recognizer_net: Pre-trained OpenCV DNN face recognition network.

    Returns:
        np.ndarray: 128-D normalized embedding vector, or None if invalid.
    """
    if face_img is None or face_img.size == 0 or recognizer_net is None:
        return None

    # SFace expects 112x112 input
    if (face_img.shape[1], face_img.shape[0]) != (112, 112):
        face_img = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_AREA)

    # Create 4D blob: swapRB=True converts BGR to RGB (SFace training format)
    blob = cv2.dnn.blobFromImage(
        face_img,
        scalefactor=1.0,
        size=(112, 112),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )
    recognizer_net.setInput(blob)
    raw_embedding = recognizer_net.forward()  # Shape: (1, 128)

    # Flatten to 1D vector
    embedding = raw_embedding.flatten().astype(np.float32)

    # L2 Normalization: embedding / ||embedding||
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def generate_dataset_embeddings(
    dataset_dir: str = "data/faces",
    recognizer_net = None,
    model_path: str = SFACE_MODEL_PATH,
    outlier_threshold: float = 0.45
) -> dict:
    """
    Generate prototype and multi-sample embeddings for all registered individuals.

    For each person:
    - Extracts 128-D normalized embeddings from each registered face sample.
    - If 4 or more samples are available, filters out outlier captures (e.g. overexposed,
      motion-blurred, or non-face frames) whose similarity to the preliminary centroid is
      below outlier_threshold (0.45).
    - Saves high-quality sample embeddings as a matrix (shape: K x 128) to capture natural
      variations in expression, lighting, and pose without noise pollution.
    - Computes and L2-normalizes the centroid mean vector.

    Returns:
        dict: {person_name: {"samples": np.ndarray(K, 128), "mean": np.ndarray(128,), "count": K, "total_samples": N}}
    """
    if recognizer_net is None:
        recognizer_net = load_face_recognizer(model_path)

    embeddings = {}
    valid_exts = {".jpg", ".jpeg", ".png"}

    if not os.path.exists(dataset_dir):
        return embeddings

    for entry in sorted(os.scandir(dataset_dir), key=lambda e: e.name.lower()):
        if entry.is_dir():
            person_name = entry.name
            raw_embeddings = []

            for file_entry in sorted(os.scandir(entry.path), key=lambda f: f.name):
                if file_entry.is_file() and os.path.splitext(file_entry.name)[1].lower() in valid_exts:
                    img = cv2.imread(file_entry.path)
                    if img is not None and img.size > 0:
                        emb = get_face_embedding(img, recognizer_net)
                        if emb is not None and emb.shape == (128,):
                            raw_embeddings.append(emb)

            if raw_embeddings:
                raw_matrix = np.array(raw_embeddings, dtype=np.float32)

                # Filter outlier samples if we have 4 or more samples
                if len(raw_embeddings) >= 4:
                    prelim_mean = np.mean(raw_matrix, axis=0)
                    prelim_norm = np.linalg.norm(prelim_mean)
                    if prelim_norm > 0:
                        prelim_mean /= prelim_norm
                        sims = np.dot(raw_matrix, prelim_mean)
                        clean_samples = [emb for emb, sim in zip(raw_embeddings, sims) if sim >= outlier_threshold]
                        if len(clean_samples) >= 2:
                            samples_matrix = np.array(clean_samples, dtype=np.float32)
                        else:
                            samples_matrix = raw_matrix
                    else:
                        samples_matrix = raw_matrix
                else:
                    samples_matrix = raw_matrix

                final_mean = np.mean(samples_matrix, axis=0)
                norm = np.linalg.norm(final_mean)
                if norm > 0:
                    final_mean /= norm

                embeddings[person_name] = {
                    "samples": samples_matrix,
                    "mean": final_mean,
                    "count": len(samples_matrix),
                    "total_samples": len(raw_embeddings)
                }

    return embeddings


def save_embeddings(embeddings: dict, cache_path: str = EMBEDDINGS_CACHE_PATH):
    """
    Cache registered embeddings to disk.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)


def load_cached_embeddings(cache_path: str = EMBEDDINGS_CACHE_PATH) -> dict:
    """
    Load cached face embeddings from disk if available.
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            return {}
    return {}


def recognize_face(
    face_crop: np.ndarray,
    recognizer_net,
    registered_embeddings: dict,
    threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN
) -> tuple:
    """
    Compare a detected face crop with registered face embeddings using calibrated cosine matching.

    Matching Process & Decision Rules:
    1. Extracts 128-D L2-normalized embedding for the live query face crop.
    2. For each registered person:
       - Computes cosine similarity against all stored high-quality samples.
       - Takes the average of the top-k (k<=3) sample similarities.
       - Computes similarity against the person's centroid prototype vector.
       - Combines both: score = 0.5 * top_samples_sim + 0.5 * centroid_sim.
    3. Identifies the best-matching candidate and second-best candidate (if available).
    4. Evaluates decision criteria:
       - If best_score < threshold:
             Decision: "Unknown Person" (score below calibrated recognition threshold).
       - If multiple registered candidates exist and (best_score - second_score) < ambiguity_margin:
             Decision: "Unknown Person" (ambiguous match, insufficient separation between candidates).
       - If best_score >= threshold and separation is sufficient:
             Decision: best_name (verified identity).

    Args:
        face_crop (np.ndarray): Detected face image crop.
        recognizer_net: Pretrained SFace network.
        registered_embeddings (dict): Output of generate_dataset_embeddings or load_cached_embeddings.
        threshold (float): Minimum cosine similarity to accept a match (default: 0.50).
        ambiguity_margin (float): Minimum separation required between top two candidates (default: 0.08).

    Returns:
        tuple: (final_name: str, confidence: float, is_known: bool, candidate_scores: dict, debug_info: dict)
    """
    empty_debug = {
        "best_name": "None",
        "best_score": 0.0,
        "second_name": None,
        "second_score": 0.0,
        "margin": 0.0,
        "is_ambiguous": False,
        "threshold": threshold,
        "final_decision": "Unknown Person",
        "is_known": False,
        "reason": "No registered profiles or invalid query input",
        "candidate_scores": {}
    }
    if not registered_embeddings or recognizer_net is None or face_crop is None:
        return "Unknown Person", 0.0, False, {}, empty_debug

    query_emb = get_face_embedding(face_crop, recognizer_net)
    if query_emb is None:
        empty_debug["reason"] = "Failed to extract face embedding from crop"
        return "Unknown Person", 0.0, False, {}, empty_debug

    candidate_scores = {}

    for name, person_data in registered_embeddings.items():
        if isinstance(person_data, dict):
            # Compute top-k sample similarity
            if "samples" in person_data and len(person_data["samples"]) > 0:
                samples_matrix = person_data["samples"]
                sample_sims = np.dot(samples_matrix, query_emb)
                k = min(3, len(sample_sims))
                top_sim = float(np.mean(np.sort(sample_sims)[-k:]))
            else:
                top_sim = None

            # Compute centroid similarity
            if "mean" in person_data and person_data["mean"] is not None:
                centroid_sim = float(np.dot(person_data["mean"], query_emb))
            else:
                centroid_sim = None

            # Combined score: 50% top samples + 50% centroid
            if top_sim is not None and centroid_sim is not None:
                score = 0.5 * top_sim + 0.5 * centroid_sim
            elif top_sim is not None:
                score = top_sim
            elif centroid_sim is not None:
                score = centroid_sim
            else:
                continue

        elif isinstance(person_data, np.ndarray) and person_data.ndim == 2:
            sample_sims = np.dot(person_data, query_emb)
            k = min(3, len(sample_sims))
            score = float(np.mean(np.sort(sample_sims)[-k:]))

        elif isinstance(person_data, np.ndarray) and person_data.ndim == 1:
            score = float(np.dot(person_data, query_emb))

        else:
            continue

        candidate_scores[name] = max(-1.0, min(1.0, float(score)))

    if not candidate_scores:
        empty_debug["reason"] = "No valid candidate scores could be computed"
        return "Unknown Person", 0.0, False, {}, empty_debug

    # Sort candidates by descending similarity score
    sorted_candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = sorted_candidates[0]

    # Check second-best candidate and ambiguity separation
    if len(sorted_candidates) >= 2:
        second_name, second_score = sorted_candidates[1]
        margin = float(best_score - second_score)
        is_ambiguous = (margin < ambiguity_margin)
    else:
        second_name = None
        second_score = 0.0
        margin = float(best_score)
        is_ambiguous = False

    # Decision logic
    if best_score < threshold:
        final_name = "Unknown Person"
        is_known = False
        reason = f"Best match ({best_name}: {best_score*100:.1f}%) < Threshold ({threshold*100:.0f}%)"
    elif is_ambiguous:
        final_name = "Unknown Person"
        is_known = False
        reason = (
            f"Ambiguous match between {best_name} ({best_score*100:.1f}%) and "
            f"{second_name} ({second_score*100:.1f}%): separation {margin*100:.1f}% < {ambiguity_margin*100:.0f}%"
        )
    else:
        final_name = best_name
        is_known = True
        reason = f"Recognized as {best_name} ({best_score*100:.1f}% >= Threshold {threshold*100:.0f}%)"

    debug_info = {
        "best_name": best_name,
        "best_score": best_score,
        "second_name": second_name,
        "second_score": second_score,
        "margin": margin,
        "is_ambiguous": is_ambiguous,
        "threshold": threshold,
        "final_decision": final_name,
        "is_known": is_known,
        "reason": reason,
        "candidate_scores": candidate_scores
    }

    return final_name, max(0.0, best_score), is_known, candidate_scores, debug_info


def draw_recognition_results(frame: np.ndarray, recognitions: list) -> np.ndarray:
    """
    Draw bounding boxes and recognition badge headers.

    Requirements:
    - Registered / Recognized face: Green box + Green badge with '{Name} ({Confidence:.1f}%)'
    - Unknown face: Red box + Red badge with 'Unknown Person'

    Args:
        frame (np.ndarray): Video frame.
        recognitions (list): List of recognition result tuples or dicts containing:
            (x, y, w, h, name, score_or_confidence, is_known, [optional label]).

    Returns:
        np.ndarray: Annotated video frame.
    """
    annotated = frame.copy()

    for item in recognitions:
        # Extract fields safely whether item is dict or tuple/list
        if isinstance(item, dict):
            box = item.get("box", (item.get("x", 0), item.get("y", 0), item.get("w", 0), item.get("h", 0)))
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            name = str(item.get("name", "Unknown Person"))
            score = float(item.get("score", item.get("confidence", 0.0)))
            is_known = bool(item.get("is_known", False))
        elif isinstance(item, (tuple, list)):
            x, y, w, h = int(item[0]), int(item[1]), int(item[2]), int(item[3])
            name = str(item[4]) if len(item) > 4 else "Unknown Person"
            score = float(item[5]) if len(item) > 5 else 0.0
            is_known = bool(item[6]) if len(item) > 6 else False
        else:
            continue

        # Convert score to percentage if passed as decimal fraction <= 1.0
        conf_pct = score * 100.0 if score <= 1.0 else score

        # Clean label construction strictly adhering to requirements:
        # - Recognized: "{Name} ({Confidence:.1f}%)"
        # - Unknown:    "Unknown Person"
        if is_known and name and name != "Unknown Person":
            display_label = f"{name} ({conf_pct:.1f}%)"
            box_color = (0, 255, 0)      # Green rectangle for recognized face
            text_color = (0, 0, 0)       # Black text on green badge for maximum legibility
        else:
            display_label = "Unknown Person"
            box_color = (0, 0, 255)      # Red rectangle for unknown face
            text_color = (255, 255, 255) # White text on red badge

        # 1. Draw bounding box around face
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

        # 2. Compute badge size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(display_label, font, font_scale, thickness)

        # 3. Position badge above bounding box (or below if too close to top edge of frame)
        if y - label_h - 12 >= 0:
            badge_y1 = y - label_h - 12
            badge_y2 = y
            text_y = y - 6
        else:
            badge_y1 = y + h
            badge_y2 = y + h + label_h + 12
            text_y = y + h + label_h + 6

        badge_x1 = x
        badge_x2 = max(x + w, x + label_w + 14)

        # 4. Draw filled background badge
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, cv2.FILLED)

        # 5. Draw label text inside badge
        cv2.putText(
            annotated,
            display_label,
            (x + 6, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

    return annotated

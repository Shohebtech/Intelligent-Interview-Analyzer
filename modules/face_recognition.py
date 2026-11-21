"""
Face Recognition Module using OpenCV SFace Deep Face Embedding Model.
Extracts 128-D facial feature embeddings and compares them using cosine similarity.
"""
import os
import cv2
import numpy as np
from config import BASE_DIR, Config
from modules.face_detection import get_face_detector

MODELS_DIR = os.path.join(BASE_DIR, 'models')
SFACE_PATH = os.path.join(MODELS_DIR, 'face_recognition_sface_2021dec.onnx')

class FaceRecognizer:
    def __init__(self, threshold=None):
        self.threshold = threshold if threshold is not None else Config.FACE_MATCH_THRESHOLD
        self.recognizer = None
        self._init_recognizer()

    def _init_recognizer(self):
        if not os.path.exists(SFACE_PATH):
            raise FileNotFoundError(f"SFace model file not found at {SFACE_PATH}")
        self.recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

    def extract_embedding(self, frame, face_raw=None):
        """
        Extract 128-D normalized face embedding from frame and face detection.
        If face_raw is None, face detection is performed automatically.
        """
        if frame is None or frame.size == 0:
            return None, "Invalid image frame"

        if face_raw is None:
            detector = get_face_detector()
            count, faces = detector.detect_faces(frame)
            if count == 0:
                return None, "No face detected in the frame. Please look directly at the camera."
            if count > 1:
                return None, "Multiple faces detected. Please ensure only you are visible."
            face_raw = faces[0]['raw']

        try:
            # Align and crop face using 5 facial landmarks
            aligned_face = self.recognizer.alignCrop(frame, face_raw)
            # Extract 128-D feature vector
            feature = self.recognizer.feature(aligned_face)
            # Normalize embedding vector
            norm = np.linalg.norm(feature)
            if norm > 1e-6:
                feature = feature / norm
            return feature.flatten().tolist(), None
        except Exception as e:
            return None, f"Failed to extract face embedding: {str(e)}"

    def compute_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two 128-D embeddings.
        Returns a float between -1.0 and 1.0 (higher = more similar).
        """
        arr1 = np.array(emb1, dtype=np.float32).flatten()
        arr2 = np.array(emb2, dtype=np.float32).flatten()
        dot = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(dot / (norm1 * norm2))

    def verify_face(self, current_embedding, candidate_embeddings):
        """
        Compare current face embedding against candidate's stored multi-angle embeddings.
        Returns:
            is_match (bool): True if candidate identity verified
            max_sim (float): Highest cosine similarity score achieved
        """
        if not current_embedding or not candidate_embeddings:
            return False, 0.0

        max_sim = -1.0
        for stored_emb in candidate_embeddings:
            sim = self.compute_similarity(current_embedding, stored_emb)
            if sim > max_sim:
                max_sim = sim

        # SFace cosine similarity threshold: 0.363 to 0.45
        is_match = bool(max_sim >= self.threshold)
        return is_match, max_sim

# Global recognizer instance
_global_recognizer = None

def get_face_recognizer():
    global _global_recognizer
    if _global_recognizer is None:
        _global_recognizer = FaceRecognizer()
    return _global_recognizer

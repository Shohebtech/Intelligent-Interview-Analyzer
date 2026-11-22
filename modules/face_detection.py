"""
Face Detection Module using OpenCV YuNet Deep Learning Detector.
Detects faces, counts occurrences, extracts bounding boxes and key facial landmarks.
"""
import os
import cv2
import numpy as np
import base64
from config import BASE_DIR

MODELS_DIR = os.path.join(BASE_DIR, 'models')
YUNET_PATH = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx')

class FaceDetector:
    def __init__(self, conf_threshold=0.6, nms_threshold=0.3):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (320, 320)
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        if not os.path.exists(YUNET_PATH):
            raise FileNotFoundError(f"YuNet model file not found at {YUNET_PATH}")
        self.detector = cv2.FaceDetectorYN.create(
            YUNET_PATH,
            "",
            self.input_size,
            self.conf_threshold,
            self.nms_threshold,
            5000
        )

    def decode_base64_frame(self, data_url):
        """Decode base64 encoded image frame into OpenCV BGR numpy array."""
        if not data_url:
            return None
        try:
            if ',' in data_url:
                header, encoded = data_url.split(',', 1)
            else:
                encoded = data_url
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"Error decoding base64 frame: {e}")
            return None

    def detect_faces(self, frame):
        """
        Detect faces in a BGR frame.
        Returns:
            face_count (int): Number of detected faces
            faces (list of dict): Each face with 'box' [x, y, w, h], 'score', 'landmarks', 'raw'
        """
        if frame is None or frame.size == 0:
            return 0, []

        h, w = frame.shape[:2]
        # Update input size dynamically for YuNet
        self.detector.setInputSize((w, h))

        _, raw_faces = self.detector.detect(frame)
        if raw_faces is None or len(raw_faces) == 0:
            return 0, []

        faces = []
        for f in raw_faces:
            # YuNet output: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            score = float(f[-1])
            landmarks = [
                (float(f[4]), float(f[5])),   # Right eye
                (float(f[6]), float(f[7])),   # Left eye
                (float(f[8]), float(f[9])),   # Nose tip
                (float(f[10]), float(f[11])), # Right mouth corner
                (float(f[12]), float(f[13]))  # Left mouth corner
            ]
            faces.append({
                'box': [max(0, x), max(0, y), fw, fh],
                'score': score,
                'landmarks': landmarks,
                'raw': f
            })

        # Sort faces by bounding box area (largest first)
        faces.sort(key=lambda item: item['box'][2] * item['box'][3], reverse=True)
        return len(faces), faces

# Global detector instance
_global_detector = None

def get_face_detector():
    global _global_detector
    if _global_detector is None:
        _global_detector = FaceDetector()
    return _global_detector

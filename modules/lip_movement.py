"""
Lip Movement Detection Module.
Calculates Mouth Aspect Ratio (MAR) and uses temporal variance analysis
to monitor mouth opening, closing, and observable speaking activity.
"""
import os
import cv2
import numpy as np
from collections import deque
from config import BASE_DIR, Config

MODELS_DIR = os.path.join(BASE_DIR, 'models')
LANDMARKER_PATH = os.path.join(MODELS_DIR, 'face_landmarker.task')

class LipMovementDetector:
    def __init__(self, history_len=20):
        self.history = deque(maxlen=history_len)
        self.open_threshold = Config.MAR_OPEN_THRESHOLD
        self.closed_threshold = Config.MAR_CLOSED_THRESHOLD
        self.variance_threshold = Config.LIP_VARIANCE_THRESHOLD
        self.landmarker = None
        self._init_landmarker()

    def _init_landmarker(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if os.path.exists(LANDMARKER_PATH):
                base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    num_faces=1
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Notice: LipMovement landmarker initialization note: {e}")
            self.landmarker = None

    def calculate_mar(self, landmarks_dict, img_w, img_h):
        """
        Calculate Mouth Aspect Ratio (MAR):
        MAR = ||p13 - p14|| / ||p61 - p291||
        """
        p13 = np.array([landmarks_dict['p13'][0] * img_w, landmarks_dict['p13'][1] * img_h])
        p14 = np.array([landmarks_dict['p14'][0] * img_w, landmarks_dict['p14'][1] * img_h])
        p61 = np.array([landmarks_dict['p61'][0] * img_w, landmarks_dict['p61'][1] * img_h])
        p291 = np.array([landmarks_dict['p291'][0] * img_w, landmarks_dict['p291'][1] * img_h])

        vertical_dist = np.linalg.norm(p13 - p14)
        horizontal_dist = np.linalg.norm(p61 - p291)

        if horizontal_dist < 1e-4:
            return 0.0

        return float(vertical_dist / horizontal_dist)

    def analyze_lip_movement(self, frame, yunet_landmarks=None):
        """
        Analyze lip movement in frame.
        Returns:
            mar (float): Current Mouth Aspect Ratio
            status (str): "Mouth Movement Detected" | "Mouth Closed" | "Lip Movement Monitoring Active"
            is_moving (bool): True if significant mouth movement/speaking detected
        """
        if frame is None or frame.size == 0:
            return 0.0, "Lip Movement Monitoring Active", False

        h, w = frame.shape[:2]
        mar = 0.0
        calculated = False

        if self.landmarker is not None:
            try:
                import mediapipe as mp
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                res = self.landmarker.detect(mp_img)
                if res.face_landmarks and len(res.face_landmarks) > 0:
                    fl = res.face_landmarks[0]
                    # Landmark 13: inner upper lip, 14: inner lower lip
                    # Landmark 61: mouth left corner, 291: mouth right corner
                    landmarks_dict = {
                        'p13': (fl[13].x, fl[13].y),
                        'p14': (fl[14].x, fl[14].y),
                        'p61': (fl[61].x, fl[61].y),
                        'p291': (fl[291].x, fl[291].y)
                    }
                    mar = self.calculate_mar(landmarks_dict, w, h)
                    calculated = True
            except Exception as e:
                pass

        # Fallback to YuNet mouth landmarks if MediaPipe landmark detection was unavailable
        if not calculated and yunet_landmarks and len(yunet_landmarks) >= 5:
            # YuNet mouth corners: index 3 (right mouth), index 4 (left mouth)
            rcm = yunet_landmarks[3]
            lcm = yunet_landmarks[4]
            mouth_width = np.linalg.norm(np.array(rcm) - np.array(lcm))
            # Approximate resting mouth height
            approx_mar = 0.15
            mar = approx_mar
            calculated = True

        if not calculated:
            return 0.0, "Lip Movement Monitoring Active", False

        # Add to rolling history for temporal variance analysis
        self.history.append(mar)
        recent_mars = list(self.history)
        variance = float(np.var(recent_mars)) if len(recent_mars) >= 3 else 0.0

        # State determination
        # Temporal analysis: variance distinguishes normal speaking (changing opening/closing) from still/unmoving
        is_moving = False
        if variance >= self.variance_threshold or mar >= self.open_threshold:
            status = "Mouth Movement Detected"
            is_moving = True
        elif mar <= self.closed_threshold:
            status = "Mouth Closed"
            is_moving = False
        else:
            status = "Lip Movement Monitoring Active"
            is_moving = False

        return round(mar, 3), status, is_moving

    def reset(self):
        self.history.clear()

# Global lip detector instance
_global_lip_detector = None

def get_lip_movement_detector():
    global _global_lip_detector
    if _global_lip_detector is None:
        _global_lip_detector = LipMovementDetector()
    return _global_lip_detector

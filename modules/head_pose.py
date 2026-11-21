"""
Head Pose and Face Rotation Estimation Module.
Uses 3D facial landmarks and solvePnP to estimate Pitch, Yaw, and Roll Euler angles.
Detects head turns (left, right, up, down) with duration filtering to prevent false positives.
"""
import os
import cv2
import numpy as np
import time
from config import BASE_DIR, Config

MODELS_DIR = os.path.join(BASE_DIR, 'models')
LANDMARKER_PATH = os.path.join(MODELS_DIR, 'face_landmarker.task')

# Standard 3D facial model coordinates (in millimeters)
FACE_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),        # Chin (landmark 152)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (landmark 33)
    (225.0, 170.0, -135.0),      # Right eye outer corner (landmark 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (landmark 291)
], dtype=np.float64)

class HeadPoseEstimator:
    def __init__(self, yaw_threshold=None, pitch_threshold=None, duration_threshold=None):
        self.yaw_threshold = yaw_threshold if yaw_threshold is not None else Config.HEAD_YAW_THRESHOLD
        self.pitch_threshold = pitch_threshold if pitch_threshold is not None else Config.HEAD_PITCH_THRESHOLD
        self.duration_threshold = duration_threshold if duration_threshold is not None else Config.HEAD_ROTATION_DURATION_WARN
        
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
                    output_facial_transformation_matrixes=True,
                    num_faces=1
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Notice: MediaPipe landmarker initialization note: {e}")
            self.landmarker = None

    def estimate_pose_from_landmarks(self, landmarks_2d, img_w, img_h):
        """
        Estimate 3D head pose from 6 key 2D landmarks using solvePnP.
        landmarks_2d: list of 6 (x, y) tuples in pixel coordinates
        """
        image_points = np.array(landmarks_2d, dtype=np.float64)
        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            FACE_MODEL_3D, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        
        # Normalize pitch around 0 degrees
        if pitch > 90:
            pitch = pitch - 180
        elif pitch < -90:
            pitch = pitch + 180

        return float(pitch), float(yaw), float(roll)

    def analyze_head_pose(self, frame, yunet_landmarks=None):
        """
        Analyze frame for head pose.
        Returns:
            pitch (float), yaw (float), roll (float), status_text (str), is_warning (bool), warning_msg (str)
        """
        if frame is None or frame.size == 0:
            return 0.0, 0.0, 0.0, "No Frame", False, None

        h, w = frame.shape[:2]
        pitch, yaw, roll = 0.0, 0.0, 0.0
        used_pnp = False

        # Method 1: MediaPipe 478 landmarks (high precision)
        if self.landmarker is not None:
            try:
                import mediapipe as mp
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                res = self.landmarker.detect(mp_img)
                if res.face_landmarks and len(res.face_landmarks) > 0:
                    fl = res.face_landmarks[0]
                    # Indices: 1 (nose), 152 (chin), 33 (left eye), 263 (right eye), 61 (left mouth), 291 (right mouth)
                    idx_list = [1, 152, 33, 263, 61, 291]
                    pts = [(fl[i].x * w, fl[i].y * h) for i in idx_list]
                    pitch, yaw, roll = self.estimate_pose_from_landmarks(pts, w, h)
                    used_pnp = True
            except Exception as e:
                pass

        # Method 2: Fallback to YuNet 5 landmarks
        if not used_pnp and yunet_landmarks and len(yunet_landmarks) >= 5:
            # YuNet: right eye, left eye, nose, right mouth, left mouth
            # Model points order: nose, chin, left eye, right eye, left mouth, right mouth
            re, le, nt, rcm, lcm = yunet_landmarks[:5]
            # Approximate chin from nose and mouth midpoint
            chin_x = nt[0]
            chin_y = ((rcm[1] + lcm[1]) / 2.0) + abs(rcm[1] - nt[1]) * 0.8
            pts = [nt, (chin_x, chin_y), le, re, lcm, rcm]
            pitch, yaw, roll = self.estimate_pose_from_landmarks(pts, w, h)
            used_pnp = True

        if not used_pnp:
            return 0.0, 0.0, 0.0, "Normal Position", False, None

        # Determine head pose status
        status = "Normal Position"
        is_warning = False
        warning_msg = None

        if yaw > self.yaw_threshold:
            status = "Face Turned Left"
            is_warning = True
            warning_msg = "Warning 1: Please Keep Your Face Towards the Camera."
        elif yaw < -self.yaw_threshold:
            status = "Face Turned Right"
            is_warning = True
            warning_msg = "Warning 1: Please Keep Your Face Towards the Camera."
        elif pitch > self.pitch_threshold:
            status = "Face Tilted Upward"
            is_warning = True
            warning_msg = "Please Maintain Proper Camera Position."
        elif pitch < -self.pitch_threshold:
            status = "Face Tilted Downward"
            is_warning = True
            warning_msg = "Please Maintain Proper Camera Position."

        return pitch, yaw, roll, status, is_warning, warning_msg

# Global estimator instance
_global_pose_estimator = None

def get_head_pose_estimator():
    global _global_pose_estimator
    if _global_pose_estimator is None:
        _global_pose_estimator = HeadPoseEstimator()
    return _global_pose_estimator

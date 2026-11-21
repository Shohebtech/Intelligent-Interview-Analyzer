# Intelligent Interview Analyzer Configuration
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
os.makedirs(INSTANCE_DIR, exist_ok=True)

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'interview-analyzer-super-secret-key-2026')
    DATABASE = os.path.join(INSTANCE_DIR, 'interview_analyzer.db')
    
    # Facial Recognition Settings
    # Distance threshold for face verification (cosine distance or Euclidean distance)
    # Lower value means stricter match
    FACE_MATCH_THRESHOLD = 0.45  # Cosine distance threshold for SFace or normalized embedding
    
    # Head Pose Thresholds (in degrees)
    HEAD_YAW_THRESHOLD = 25.0    # Left / Right rotation limit
    HEAD_PITCH_THRESHOLD = 20.0  # Up / Down tilt limit
    HEAD_ROTATION_DURATION_WARN = 2.0 # Seconds before persistent warning
    
    # Lip Movement / Mouth Aspect Ratio (MAR) Thresholds
    MAR_OPEN_THRESHOLD = 0.45    # MAR above which mouth is considered distinctly open
    MAR_CLOSED_THRESHOLD = 0.25  # MAR below which mouth is considered closed
    LIP_VARIANCE_THRESHOLD = 0.003 # Variance in recent window indicating speaking movement
    
    # Security & Termination Policies
    IDENTITY_MISMATCH_GRACE_PERIOD = 5.0 # Seconds allowed to restore identity before auto-termination
    NO_FACE_WARN_THRESHOLD = 7.0         # Seconds without face before security escalation
    MULTIPLE_FACES_GRACE_PERIOD = 10.0   # Seconds of multiple faces before termination
    
    # Server configuration
    HOST = '0.0.0.0'
    PORT = 8000
    DEBUG = False

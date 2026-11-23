# Intelligent Interview Analyzer Configuration
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
os.makedirs(INSTANCE_DIR, exist_ok=True)

# Automatically load .env file if present
env_file = os.path.join(BASE_DIR, '.env')
if os.path.exists(env_file):
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception as e:
        print(f"Notice loading .env: {e}")

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'interview-analyzer-super-secret-key-2026')
    DATABASE = os.path.join(INSTANCE_DIR, 'interview_analyzer.db')
    
    # Session Security
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 86400  # 24 hours
    
    # Email / SMTP Configuration (set via environment variables or .env)
    SMTP_HOST = os.environ.get('SMTP_HOST', '')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    SMTP_USER = os.environ.get('SMTP_USER', '')
    SMTP_PASS = os.environ.get('SMTP_PASS', '')
    SMTP_USE_TLS = os.environ.get('SMTP_USE_TLS', 'True').lower() in ('true', '1', 't')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'no-reply@interview-analyzer.com')
    
    # OTP Configuration
    OTP_EXPIRY_MINUTES = 10
    OTP_RESEND_COOLDOWN_SECONDS = 60
    
    # Password Policy
    MIN_PASSWORD_LENGTH = 12
    
    # Facial Recognition Settings
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

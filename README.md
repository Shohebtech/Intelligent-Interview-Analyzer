# Intelligent Interview Analyzer

**Intelligent Interview Analyzer** is an AI-powered interview integrity monitoring and proctoring web application. It monitors candidates during live interviews using real-time face detection, registered deep facial biometrics, 3D head-pose estimation, observable lip movement tracking, an automated 5-second identity mismatch termination system, and enterprise-grade authentication with email OTP verification, strong password generation, and bcrypt hashing.

---

## Key Features

1. **Email ID Authentication & Verification**:
   - RFC 5322 format validation during registration.
   - Duplicate email registration prevention with user-friendly error messages.
   - Mandatory email verification via 6-digit OTP dispatched to candidate's email.
   - 60-second cooldown timer for Resend OTP.
   - Only verified email addresses are permitted to log in.

2. **Strong Password Generation & Enforcement**:
   - Suggested strong passwords via one-click **"Suggest Strong"** generator (cryptographically random 14-character passwords containing uppercase, lowercase, numbers, and special characters).
   - Real-time password strength meter (**Weak**, **Medium**, **Strong**, **Very Strong**).
   - Live visual requirement checklist (12+ characters, uppercase, lowercase, digits, special characters).
   - Show/Hide Password visibility toggle.
   - Rejection of weak passwords on frontend and backend.

3. **Secure Login & Session Management**:
   - Authentication via registered Email ID or Candidate ID + Password.
   - Industry-standard **bcrypt** password hashing (work factor 12).
   - Protection against timing attacks and clear messages for invalid credentials vs. unverified accounts.
   - Secure HTTP-only session cookie management.

4. **Self-Service Forgot Password Recovery**:
   - Password reset via 6-digit email OTP.
   - Enforces 60-second cooldown timer on reset OTP requests.
   - Requires strong new password validation with confirm password matching.
   - Updates password in database using bcrypt.

5. **Webcam Candidate Biometric Registration**:
   - Multi-angle facial image enrollment (Frontal, Left, Right) directly through the browser webcam via `navigator.mediaDevices.getUserMedia()`.
   - Generates and stores 128-D normalized deep facial feature embeddings in SQLite.
   - Prevents registration without a valid, clearly detected face.
   - No hardcoded or dummy credentials.

6. **Real-Time Face Detection & Deep Biometric Recognition**:
   - **Condition A (Verified)**: Continuous verification against enrolled embeddings. Displays green **"Identity Verified"** status.
   - **Condition B (No Face)**: Displays yellow warning: *"Face Not Detected. Please Stay in Front of the Camera."*
   - **Condition C (Identity Mismatch)**: Displays prominent red alert: *"Identity Mismatch Detected! Please return to your registered position."* and initiates a strict **5-second countdown**. If the registered candidate does not return within 5 seconds, the interview **automatically terminates** on the backend and database.
   - **Condition D (Multiple Faces)**: Displays warning: *"Multiple Faces Detected. Only One Candidate Is Allowed."*

7. **3D Head-Pose & Face Rotation Estimation**:
   - Uses MediaPipe 3D facial landmarks and `cv2.solvePnP` to compute Pitch, Yaw, and Roll Euler angles.
   - Detects turning left, turning right, tilting upward, and tilting downward.
   - Employs temporal filtering to ignore natural eye blinking or minor movements while flagging sustained head turns.

8. **Observable Lip Movement Detection (MAR)**:
   - Calculates Mouth Aspect Ratio:
     $$\text{MAR} = \frac{\|p_{13} - p_{14}\|}{\|p_{61} - p_{291}\|}$$
   - Rolling variance analysis monitors observable speaking activity and states: *"Mouth Movement Detected"*, *"Mouth Closed"*, and *"Lip Movement Monitoring Active"*.

9. **Server-Side Termination Enforcement**:
   - Termination logic is enforced in Python on the backend.
   - Once terminated, session status is updated to `'terminated'` in SQLite, the termination reason and timestamp are locked, and answer submissions are rejected (HTTP 403).

10. **Administrator Console**:
    - Initialize the first administrator account (no default credentials).
    - Review registered candidates and biometric enrollment status.
    - Review interview histories, warning counts, timestamps, and termination reasons.
    - Inspect full audit trail of monitoring events and submitted answers.
    - Delete candidate records with cascading cleanup.

---

## Technology Stack

- **Backend**: Python 3.10+, Flask, OpenCV (`cv2.FaceDetectorYN`, `cv2.FaceRecognizerSF`, `solvePnP`), MediaPipe Tasks (`FaceLandmarker`), NumPy, SQLite3, bcrypt.
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, FontAwesome 6.
- **Biometric Models**:
  - YuNet Face Detector (`face_detection_yunet_2023mar.onnx`)
  - SFace Deep Face Recognition (`face_recognition_sface_2021dec.onnx`)
  - MediaPipe Face Landmarker (`face_landmarker.task`)

---

## Directory Structure

```text
IntelligentInterviewAnalyzer/
│
├── server.py                # Main Flask application entry point
├── config.py                # Application, security, SMTP & threshold configurations
├── requirements.txt         # Python package dependencies (Flask, OpenCV, MediaPipe, bcrypt)
├── README.md                # Documentation and execution guide
│
├── database/
│   └── db.py                # SQLite schema, bcrypt hashing, migrations, queries, and OTP storage
│
├── models/
│   ├── face_landmarker.task             # MediaPipe 3D face mesh landmarker
│   ├── face_detection_yunet_2023mar.onnx # OpenCV YuNet deep face detector
│   └── face_recognition_sface_2021dec.onnx # OpenCV SFace deep face recognizer
│
├── modules/
│   ├── auth_utils.py        # Email validation, OTP dispatch, strong password generator & evaluator
│   ├── face_detection.py    # Frame decoding, face bounding boxes & landmarks
│   ├── face_recognition.py  # 128-D embedding extraction & cosine matching
│   ├── head_pose.py         # 3D pose estimation (Pitch, Yaw, Roll)
│   ├── lip_movement.py      # Mouth Aspect Ratio (MAR) & variance analysis
│   └── interview_monitor.py # Proctor state machine & 5s auto-termination
│
├── routes/
│   ├── auth.py              # Candidate & Admin registration, OTP, login, forgot-password, logout
│   ├── candidate.py         # Candidate dashboard & past interview records
│   ├── interview.py         # Live interview session, frame evaluation, answers
│   └── admin.py             # Administrator audit console & candidate deletion
│
├── templates/
│   ├── base.html            # Dark navy theme master layout
│   ├── index.html           # Landing page with feature highlights
│   ├── register.html        # Candidate registration with email OTP & webcam face enrollment
│   ├── login.html           # Candidate sign-in with show/hide password and forgot password link
│   ├── forgot_password.html # Self-service password recovery with OTP & strong password reset
│   ├── dashboard.html       # Candidate portal & interview launcher
│   ├── interview.html       # Live interview HUD, proctor stream & questions
│   ├── terminated.html      # Official interview termination page
│   ├── admin_login.html     # Admin sign-in / initial admin setup
│   └── admin.html           # Admin audit dashboard
│
├── static/
│   └── css/
│       └── style.css        # Dark navy, white cards, purple/blue accents
│
├── tests/
│   └── test_analyzer.py     # Automated unit & integration test suite (18 test scenarios)
│
└── instance/                # SQLite database and temporary cache storage
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Modern web browser (Chrome, Edge, Firefox) with camera permissions enabled
- Working webcam

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables (Optional for SMTP)
Configure environment variables if you want to use external SMTP for sending verification emails (otherwise, codes are logged to console and `instance/otp_dev_log.json` automatically for seamless local development):
```bash
# Optional SMTP configuration
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT=587
export SMTP_USER="your-email@gmail.com"
export SMTP_PASS="your-app-password"
export MAIL_DEFAULT_SENDER="no-reply@interview-analyzer.com"
```

---

## Running the Application

Launch the local server:

```bash
python server.py
```

Open your browser and navigate to:

```text
http://localhost:8000
```

---

## Verification & Testing Scenarios

Run the complete 18-scenario automated test suite:

```bash
python -m unittest tests/test_analyzer.py
```

### Testing Scenarios Covered:
1. **Registration with valid email**: RFC 5322 format validation.
2. **Registration with invalid email**: Rejection of malformed email addresses.
3. **OTP verification**: 6-digit OTP creation, verification, cooldown enforcement, and expired/used code rejection.
4. **Registration using existing email**: Duplicate email prevention.
5. **Weak password rejection**: Enforces length >= 12, uppercase, lowercase, numbers, and special characters.
6. **Strong password generation**: Cryptographically random 14-character passwords satisfying all criteria.
7. **Login with correct credentials**: Validates against database with bcrypt.
8. **Login with incorrect credentials**: Rejects bad passwords or unregistered emails.
9. **Forgot password functionality**: OTP dispatch, verification, bcrypt password update, and login with new password.
10. **Logout functionality**: Secure session clearance.
11. **Database initialization**: Tables, foreign keys, schema migrations.
12. **Candidate registration & embeddings**: Biometric enrollment.
13. **Face recognition cosine matching**: Threshold verification.
14. **Head pose estimation**: 3D solvePnP Euler angles.
15. **Lip movement MAR**: Mouth opening/closing detection.
16. **5-second identity mismatch auto-termination**: Automatic termination on backend.
17. **Identity restoration**: Cancels countdown when registered candidate returns within 5 seconds.
18. **Admin management & candidate deletion**: Administrator console and cascading deletion.

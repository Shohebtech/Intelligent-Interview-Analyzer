<<<<<<< HEAD
# Intelligent-Interview-Analyzer
=======
# Intelligent Interview Analyzer

**Intelligent Interview Analyzer** is an AI-powered interview integrity monitoring and proctoring web application. It monitors candidates during live interviews using real-time face detection, registered deep facial biometrics, 3D head-pose estimation, observable lip movement tracking, and an automated 5-second identity mismatch termination system.

---

## Key Features

1. **Webcam Candidate Biometric Registration**:
   - Multi-angle facial image enrollment (Frontal, Left, Right) directly through the browser webcam via `navigator.mediaDevices.getUserMedia()`.
   - Generates and stores 128-D normalized deep facial feature embeddings in SQLite.
   - Prevents registration without a valid, clearly detected face.
   - No hardcoded or dummy credentials.

2. **Real-Time Face Detection & Deep Biometric Recognition**:
   - **Condition A (Verified)**: Continuous verification against enrolled embeddings. Displays green **"Identity Verified"** status.
   - **Condition B (No Face)**: Displays yellow warning: *"Face Not Detected. Please Stay in Front of the Camera."*
   - **Condition C (Identity Mismatch)**: Displays prominent red alert: *"Identity Mismatch Detected! Please return to your registered position."* and initiates a strict **5-second countdown**. If the registered candidate does not return within 5 seconds, the interview **automatically terminates** on the backend and database.
   - **Condition D (Multiple Faces)**: Displays warning: *"Multiple Faces Detected. Only One Candidate Is Allowed."*

3. **3D Head-Pose & Face Rotation Estimation**:
   - Uses MediaPipe 3D facial landmarks and `cv2.solvePnP` to compute Pitch, Yaw, and Roll Euler angles.
   - Detects turning left, turning right, tilting upward, and tilting downward.
   - Employs temporal filtering to ignore natural eye blinking or minor movements while flagging sustained head turns.

4. **Observable Lip Movement Detection (MAR)**:
   - Calculates Mouth Aspect Ratio:
     $$\text{MAR} = \frac{\|p_{13} - p_{14}\|}{\|p_{61} - p_{291}\|}$$
   - Rolling variance analysis monitors observable speaking activity and states: *"Mouth Movement Detected"*, *"Mouth Closed"*, and *"Lip Movement Monitoring Active"*.

5. **Server-Side Termination Enforcement**:
   - Termination logic is enforced in Python on the backend.
   - Once terminated, session status is updated to `'terminated'` in SQLite, the termination reason and timestamp are locked, and answer submissions are rejected (HTTP 403).

6. **Administrator Console**:
   - Initialize the first administrator account (no default credentials).
   - Review registered candidates and biometric enrollment status.
   - Review interview histories, warning counts, timestamps, and termination reasons.
   - Inspect full audit trail of monitoring events and submitted answers.
   - Delete candidate records with cascading cleanup.

---

## Technology Stack

- **Backend**: Python 3.10+, Flask, OpenCV (`cv2.FaceDetectorYN`, `cv2.FaceRecognizerSF`, `solvePnP`), MediaPipe Tasks (`FaceLandmarker`), NumPy, SQLite3, Werkzeug.
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
├── config.py                # Application and threshold configurations
├── requirements.txt         # Python package dependencies
├── README.md                # Documentation and execution guide
│
├── database/
│   └── db.py                # SQLite schema, connection management, queries
│
├── models/
│   ├── face_landmarker.task             # MediaPipe 3D face mesh landmarker
│   ├── face_detection_yunet_2023mar.onnx # OpenCV YuNet deep face detector
│   └── face_recognition_sface_2021dec.onnx # OpenCV SFace deep face recognizer
│
├── modules/
│   ├── face_detection.py    # Frame decoding, face bounding boxes & landmarks
│   ├── face_recognition.py  # 128-D embedding extraction & cosine matching
│   ├── head_pose.py         # 3D pose estimation (Pitch, Yaw, Roll)
│   ├── lip_movement.py      # Mouth Aspect Ratio (MAR) & variance analysis
│   └── interview_monitor.py # Proctor state machine & 5s auto-termination
│
├── routes/
│   ├── auth.py              # Candidate & Admin registration / login / logout
│   ├── candidate.py         # Candidate dashboard & past interview records
│   ├── interview.py         # Live interview session, frame evaluation, answers
│   └── admin.py             # Administrator audit console & candidate deletion
│
├── templates/
│   ├── base.html            # Dark navy theme master layout
│   ├── index.html           # Landing page with feature highlights
│   ├── register.html        # Multi-angle webcam face enrollment wizard
│   ├── login.html           # Candidate sign-in
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
│   └── test_analyzer.py     # Automated unit & integration test suite
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

### 3. Model Weights
The pre-trained vision models are located in the `models/` directory:
- `models/face_landmarker.task`
- `models/face_detection_yunet_2023mar.onnx`
- `models/face_recognition_sface_2021dec.onnx`

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

## User Workflows

### Candidate Flow:
1. Navigate to `http://localhost:8000/register`.
2. Enter Full Name, Candidate ID, Email, and Password.
3. Allow browser camera access.
4. Capture face images (Frontal is required; Left and Right angles are recommended).
5. Click **"Register & Save Biometric Profile"**.
6. Sign in at `http://localhost:8000/login` with your Candidate ID and password.
7. Click **"Start New Interview"** on the dashboard.
8. Answer the questions while maintaining normal camera position.

### Administrator Flow:
1. Navigate to `http://localhost:8000/admin/login`.
2. On first visit, initialize the administrator account with your desired username and password.
3. View all registered candidates, interview outcomes, warning counts, and termination reasons.
4. Click **"Events"** on any session to view the full audit trail and candidate answers.

---

## Verification & Testing Scenarios

Run the automated test suite:

```bash
python -m unittest tests/test_analyzer.py
```

### Physical Webcam Testing Matrix (Section 13):

| Scenario | Action | Expected Application Behavior |
|---|---|---|
| **TEST 1** | Registered candidate enters interview. | Status indicator displays **"Identity Verified"** (Green). |
| **TEST 2** | Candidate moves face slightly / blinks. | Natural micro-movements ignored; **no false warning**. |
| **TEST 3** | Candidate rotates face significantly (>25° Yaw) left or right. | Yellow warning displays: *"Warning 1: Please Keep Your Face Towards the Camera."* |
| **TEST 4** | An unregistered person enters the frame. | Status turns Red: *"Identity Mismatch Detected! Please return to your registered position."* and **5-second countdown** begins. |
| **TEST 5** | Unregistered person remains in frame for > 5 seconds. | Session **automatically terminates**. Candidate is redirected to `/interview/terminated`. No further answers accepted. |
| **TEST 6** | Registered candidate returns to frame within 5 seconds. | Countdown is **cancelled immediately**, green *"Identity Verified"* restores, interview continues. |
| **TEST 7** | Candidate moves lips while speaking response. | Lip movement indicator shows **"Mouth Movement Detected"** (MAR active). |
| **TEST 8** | Camera is disconnected / disabled. | Camera error notification displays; candidate alerted to reconnect webcam. |
>>>>>>> master

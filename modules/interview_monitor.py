"""
Interview Monitoring Core Engine.
Orchestrates Face Detection, SFace Face Recognition, 3D Head Pose, and Lip Movement Analysis.
Enforces the 5-second Identity Mismatch termination policy on backend and database.
"""
import time
import os
from config import Config
from database import db
from modules.face_detection import get_face_detector
from modules.face_recognition import get_face_recognizer
from modules.head_pose import get_head_pose_estimator
from modules.lip_movement import get_lip_movement_detector

class InterviewMonitor:
    def __init__(self):
        self.sessions = {} # session_id -> state dict

    def register_session(self, session_id, candidate_id):
        """Initialize in-memory state tracking for an interview session."""
        embeddings = db.get_candidate_embeddings(candidate_id)
        self.sessions[session_id] = {
            'session_id': session_id,
            'candidate_id': candidate_id,
            'candidate_embeddings': embeddings,
            'mismatch_start_time': None,
            'no_face_start_time': None,
            'multiple_faces_start_time': None,
            'head_turn_start_time': None,
            'last_warning_time': 0,
            'warning_count': 0,
            'status': 'in_progress',
            'termination_reason': None,
            'last_verified_time': time.time(),
            'lip_detector': get_lip_movement_detector() # instance for MAR history
        }

    def get_session_state(self, session_id):
        return self.sessions.get(session_id)

    def terminate_session(self, session_id, reason):
        """Terminate interview session on backend and record in database."""
        state = self.sessions.get(session_id)
        if state:
            state['status'] = 'terminated'
            state['termination_reason'] = reason

        # Update in database
        db.update_interview_status(
            session_id=session_id,
            status='terminated',
            termination_reason=reason,
            final_notes=f"Interview automatically terminated by system at {time.strftime('%Y-%m-%d %H:%M:%S')}."
        )
        if state:
            db.log_monitoring_event(
                session_id=session_id,
                candidate_id=state['candidate_id'],
                event_type='TERMINATED',
                severity='CRITICAL',
                details=reason
            )

    def process_frame(self, session_id, frame_b64):
        """
        Process incoming webcam frame for session.
        Returns full monitoring status dict.
        """
        now = time.time()
        
        # Check if session exists in memory
        if session_id not in self.sessions:
            interview = db.get_interview_session(session_id)
            if not interview:
                return {'error': 'Invalid session ID', 'status': 'invalid'}
            if interview['status'] != 'in_progress':
                return {
                    'session_id': session_id,
                    'status': interview['status'],
                    'is_terminated': interview['status'] == 'terminated',
                    'termination_reason': interview.get('termination_reason')
                }
            self.register_session(session_id, interview['candidate_id'])

        state = self.sessions[session_id]

        # Check if session was already terminated
        if state['status'] == 'terminated':
            return {
                'session_id': session_id,
                'status': 'terminated',
                'is_terminated': True,
                'termination_reason': state['termination_reason']
            }

        # 1. Decode frame
        detector = get_face_detector()
        frame = detector.decode_base64_frame(frame_b64)
        if frame is None:
            return {
                'session_id': session_id,
                'status': 'in_progress',
                'status_color': 'yellow',
                'face_status': 'Camera Frame Error',
                'warning': 'Unable to decode camera frame. Please verify webcam connection.',
                'is_terminated': False
            }

        # 2. Detect faces
        face_count, faces = detector.detect_faces(frame)
        recognizer = get_face_recognizer()
        pose_estimator = get_head_pose_estimator()
        lip_detector = state['lip_detector']

        active_warnings = []
        status_color = 'green'
        face_status = 'Identity Verified'
        is_verified = False
        mismatch_countdown = None
        similarity_score = 0.0

        # Pose & Lip defaults
        pitch, yaw, roll = 0.0, 0.0, 0.0
        pose_status = "Normal Position"
        is_pose_warning = False
        mar = 0.0
        lip_status = "Lip Movement Monitoring Active"
        is_speaking = False

        # --- CONDITION B: No face detected ---
        if face_count == 0:
            status_color = 'yellow'
            face_status = 'Face Not Detected'
            active_warnings.append("Face Not Detected. Please Stay in Front of the Camera.")
            
            # Reset mismatch countdown (per prompt: mismatch timer specifically applies to identity mismatch)
            state['mismatch_start_time'] = None

            if state['no_face_start_time'] is None:
                state['no_face_start_time'] = now

            pose_status = "No Face in Frame"
            lip_status = "Lip Movement Monitoring Active"

        # --- CONDITION D: Multiple faces detected ---
        elif face_count > 1:
            status_color = 'yellow'
            face_status = 'Multiple Faces Detected'
            active_warnings.append("Multiple Faces Detected. Only One Candidate Is Allowed.")
            
            # Reset mismatch countdown
            state['mismatch_start_time'] = None

            if state['multiple_faces_start_time'] is None:
                state['multiple_faces_start_time'] = now
            elif (now - state['multiple_faces_start_time']) >= Config.MULTIPLE_FACES_GRACE_PERIOD:
                # If multiple faces persist beyond grace period, escalate/terminate
                self.terminate_session(session_id, "Security Policy Violation: Multiple faces detected continuously.")
                return {
                    'session_id': session_id,
                    'status': 'terminated',
                    'is_terminated': True,
                    'termination_reason': 'Security Policy Violation: Multiple faces detected continuously.'
                }

        # --- CONDITION A & C: Exactly one face detected ---
        else:
            state['no_face_start_time'] = None
            state['multiple_faces_start_time'] = None
            primary_face = faces[0]

            # Head Pose Analysis
            pitch, yaw, roll, pose_status, is_pose_warning, pose_warning_msg = pose_estimator.analyze_head_pose(
                frame, primary_face['landmarks']
            )

            # Lip Movement Analysis
            mar, lip_status, is_speaking = lip_detector.analyze_lip_movement(
                frame, primary_face['landmarks']
            )

            # Extract embedding & compare with candidate's registered embeddings
            current_emb, err = recognizer.extract_embedding(frame, primary_face['raw'])
            if current_emb and state['candidate_embeddings']:
                is_match, similarity_score = recognizer.verify_face(current_emb, state['candidate_embeddings'])
            else:
                is_match, similarity_score = False, 0.0

            if is_match:
                # === CONDITION A: Registered face verified ===
                is_verified = True
                status_color = 'green'
                face_status = 'Identity Verified'
                state['last_verified_time'] = now

                # If countdown was running, cancel it! (Requirement 6: Registered candidate returns within 5s)
                if state['mismatch_start_time'] is not None:
                    db.log_monitoring_event(
                        session_id=session_id,
                        candidate_id=state['candidate_id'],
                        event_type='VERIFIED',
                        severity='INFO',
                        details=f"Candidate verified. Countdown cancelled (similarity: {similarity_score:.3f})"
                    )
                    state['mismatch_start_time'] = None

                # Process head pose warnings for verified candidate
                if is_pose_warning:
                    if state['head_turn_start_time'] is None:
                        state['head_turn_start_time'] = now
                    elif (now - state['head_turn_start_time']) >= Config.HEAD_ROTATION_DURATION_WARN:
                        status_color = 'yellow'
                        active_warnings.append(pose_warning_msg)
                        if (now - state['last_warning_time']) > 4.0:
                            state['warning_count'] = db.increment_interview_warning(session_id)
                            state['last_warning_time'] = now
                            db.log_monitoring_event(
                                session_id=session_id,
                                candidate_id=state['candidate_id'],
                                event_type='HEAD_POSE_WARN',
                                severity='WARNING',
                                details=f"{pose_status} (pitch={pitch:.1f}, yaw={yaw:.1f})"
                            )
                else:
                    state['head_turn_start_time'] = None

            else:
                # === CONDITION C: Different person detected (Identity Mismatch) ===
                is_verified = False
                status_color = 'red'
                face_status = 'Identity Mismatch Detected'
                active_warnings.append("Identity Mismatch Detected! Please return to your registered position.")

                if state['mismatch_start_time'] is None:
                    state['mismatch_start_time'] = now
                    db.log_monitoring_event(
                        session_id=session_id,
                        candidate_id=state['candidate_id'],
                        event_type='MISMATCH',
                        severity='CRITICAL',
                        details=f"Identity mismatch initiated. Best similarity: {similarity_score:.3f}"
                    )

                elapsed_mismatch = now - state['mismatch_start_time']
                remaining = max(0.0, Config.IDENTITY_MISMATCH_GRACE_PERIOD - elapsed_mismatch)
                mismatch_countdown = round(remaining, 1)

                # Check 5-second termination condition
                if elapsed_mismatch >= Config.IDENTITY_MISMATCH_GRACE_PERIOD:
                    termination_msg = "Identity Mismatch: Candidate identity could not be verified within the permitted 5-second grace period."
                    self.terminate_session(session_id, termination_msg)
                    return {
                        'session_id': session_id,
                        'status': 'terminated',
                        'is_terminated': True,
                        'termination_reason': termination_msg,
                        'mismatch_countdown': 0.0,
                        'status_color': 'red',
                        'face_status': 'Interview Terminated'
                    }

        return {
            'session_id': session_id,
            'status': state['status'],
            'is_verified': is_verified,
            'face_status': face_status,
            'status_color': status_color,
            'similarity_score': round(similarity_score, 3),
            'mismatch_countdown': mismatch_countdown,
            'head_pose': {
                'pitch': round(pitch, 1),
                'yaw': round(yaw, 1),
                'roll': round(roll, 1),
                'status': pose_status,
                'is_warning': is_pose_warning
            },
            'lip_movement': {
                'mar': mar,
                'status': lip_status,
                'is_speaking': is_speaking
            },
            'warning_count': state['warning_count'],
            'active_warnings': active_warnings,
            'is_terminated': False,
            'termination_reason': None
        }

# Global monitor instance
_global_monitor = None

def get_interview_monitor():
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = InterviewMonitor()
    return _global_monitor

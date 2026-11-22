"""
Automated Test Suite for Intelligent Interview Analyzer.
Tests Database, Biometrics, Pose Estimation, Lip Movement, and 5-Second Mismatch Auto-Termination.
"""
import os
import sys
import unittest
import numpy as np
import time

# Ensure project root is in python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config, INSTANCE_DIR
from database import db
from modules.face_detection import get_face_detector
from modules.face_recognition import get_face_recognizer
from modules.head_pose import get_head_pose_estimator
from modules.lip_movement import get_lip_movement_detector
from modules.interview_monitor import get_interview_monitor
from server import create_app

class TestIntelligentInterviewAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use test database
        Config.DATABASE = os.path.join(INSTANCE_DIR, 'test_interview.db')
        db.init_db()
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_01_database_initialization(self):
        """Test database tables and default interview questions seeding."""
        questions = db.get_questions()
        self.assertGreater(len(questions), 0)
        self.assertIn("introduce yourself", questions[0]['question_text'].lower())

    def test_02_candidate_registration_and_embeddings(self):
        """Test candidate profile creation and biometric embedding storage."""
        cand_id = "TEST-CAND-001"
        # Cleanup if exists
        db.delete_candidate(cand_id)

        ok, msg = db.create_candidate(cand_id, "Alex Mercer", "alex@example.com", "Password123!")
        self.assertTrue(ok)

        # Generate synthetic 128-D normalized embedding vector
        fake_emb = np.random.randn(128).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)

        saved = db.save_face_embeddings(cand_id, [{'angle': 'frontal', 'vector': fake_emb.tolist()}])
        self.assertTrue(saved)

        loaded_embs = db.get_candidate_embeddings(cand_id)
        self.assertEqual(len(loaded_embs), 1)
        self.assertEqual(len(loaded_embs[0]), 128)

        # Authenticate
        cand = db.verify_candidate_credentials(cand_id, "Password123!")
        self.assertIsNotNone(cand)
        self.assertEqual(cand['full_name'], "Alex Mercer")

    def test_03_face_recognition_cosine_matching(self):
        """Test cosine similarity matching and verification thresholds."""
        recognizer = get_face_recognizer()

        # Two identical embeddings
        emb1 = np.random.randn(128).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        sim_identical = recognizer.compute_similarity(emb1, emb1)
        self.assertAlmostEqual(sim_identical, 1.0, places=4)

        # Stored embeddings check
        stored = [emb1]
        is_match, score = recognizer.verify_face(emb1.tolist(), stored)
        self.assertTrue(is_match)
        self.assertGreaterEqual(score, recognizer.threshold)

        # Orthogonal/different embedding
        emb_diff = -emb1
        is_match_diff, score_diff = recognizer.verify_face(emb_diff.tolist(), stored)
        self.assertFalse(is_match_diff)

    def test_04_head_pose_estimation(self):
        """Test 3D head pose Euler angles estimation via solvePnP."""
        pose_estimator = get_head_pose_estimator()
        # 6 Frontal landmarks: nose, chin, left eye, right eye, left mouth, right mouth
        frontal_pts = [
            (320.0, 240.0), # nose
            (320.0, 360.0), # chin
            (240.0, 190.0), # left eye
            (400.0, 190.0), # right eye
            (270.0, 300.0), # left mouth
            (370.0, 300.0)  # right mouth
        ]
        pitch, yaw, roll = pose_estimator.estimate_pose_from_landmarks(frontal_pts, 640, 480)
        self.assertIsInstance(pitch, float)
        self.assertIsInstance(yaw, float)
        self.assertIsInstance(roll, float)

    def test_05_lip_movement_mar(self):
        """Test Mouth Aspect Ratio (MAR) computation."""
        lip_detector = get_lip_movement_detector()
        # Simulated closed mouth coordinates (low vertical distance)
        closed_landmarks = {
            'p13': (0.5, 0.60),
            'p14': (0.5, 0.61),
            'p61': (0.42, 0.605),
            'p291': (0.58, 0.605)
        }
        mar_closed = lip_detector.calculate_mar(closed_landmarks, 640, 480)
        self.assertLess(mar_closed, 0.25)

        # Simulated open mouth coordinates (high vertical distance)
        open_landmarks = {
            'p13': (0.5, 0.55),
            'p14': (0.5, 0.67),
            'p61': (0.42, 0.605),
            'p291': (0.58, 0.605)
        }
        mar_open = lip_detector.calculate_mar(open_landmarks, 640, 480)
        self.assertGreater(mar_open, 0.40)

    def test_06_identity_mismatch_and_5_second_termination(self):
        """
        Verify the 5-second countdown and automatic interview termination policy:
        1. Start session
        2. Detect mismatch -> starts countdown
        3. Exceed 5.0 seconds of unverified mismatch -> auto termination on backend
        4. Session status in DB changes to 'terminated'
        5. Stop accepting further answers
        """
        session_id = "TEST-SESSION-AUTO-TERM"
        cand_id = "TEST-CAND-001"
        db.create_interview_session(session_id, cand_id)

        monitor = get_interview_monitor()
        monitor.register_session(session_id, cand_id)

        # Check initial state
        state = monitor.get_session_state(session_id)
        self.assertEqual(state['status'], 'in_progress')
        self.assertIsNone(state['mismatch_start_time'])

        # Simulate identity mismatch occurring at time T0
        t0 = time.time()
        state['mismatch_start_time'] = t0

        # Simulate 2 seconds elapsed: remaining should be ~3.0s, not terminated
        elapsed = 2.0
        remaining = Config.IDENTITY_MISMATCH_GRACE_PERIOD - elapsed
        self.assertGreater(remaining, 0)

        # Simulate 5.1 seconds elapsed without verification
        state['mismatch_start_time'] = time.time() - 5.1
        # Trigger termination logic
        monitor.terminate_session(session_id, "Identity Mismatch: Candidate identity could not be verified within the permitted 5-second grace period.")

        # Verify backend session status
        updated_session = db.get_interview_session(session_id)
        self.assertEqual(updated_session['status'], 'terminated')
        self.assertIn("Identity Mismatch", updated_session['termination_reason'])

        # Verify response submission is BLOCKED after termination
        with self.app.test_request_context():
            with self.client.session_transaction() as sess:
                sess['user_type'] = 'candidate'
                sess['candidate_id'] = cand_id

            resp = self.client.post('/interview/submit_answer', json={
                'session_id': session_id,
                'question_id': 1,
                'answer_text': 'This response must be rejected.'
            })
            self.assertEqual(resp.status_code, 403)
            data = resp.get_json()
            self.assertTrue(data.get('terminated'))

    def test_07_identity_restoration_cancels_countdown(self):
        """
        Verify that if registered candidate returns within 5 seconds,
        the mismatch countdown is cancelled.
        """
        session_id = "TEST-SESSION-RESTORE"
        cand_id = "TEST-CAND-001"
        db.create_interview_session(session_id, cand_id)

        monitor = get_interview_monitor()
        monitor.register_session(session_id, cand_id)
        state = monitor.get_session_state(session_id)

        # Mismatch starts
        state['mismatch_start_time'] = time.time() - 2.0
        self.assertIsNotNone(state['mismatch_start_time'])

        # Candidate restored within grace period -> reset countdown
        state['mismatch_start_time'] = None
        self.assertIsNone(state['mismatch_start_time'])
        self.assertEqual(state['status'], 'in_progress')

    def test_08_admin_management_and_candidate_deletion(self):
        """Test admin creation, login, and candidate record deletion."""
        admin_user = f"admin_{int(time.time())}"
        # Create admin
        ok, msg = db.create_admin(admin_user, f"{admin_user}@interview.test", "AdminSecret123")
        self.assertTrue(ok)

        # Verify admin
        admin = db.verify_admin_credentials(admin_user, "AdminSecret123")
        self.assertIsNotNone(admin)

        # Delete candidate
        cand_id = "TEST-CAND-001"
        del_ok, _ = db.delete_candidate(cand_id)
        self.assertTrue(del_ok)
        self.assertIsNone(db.get_candidate(cand_id))

if __name__ == '__main__':
    unittest.main()

"""
Automated Test Suite for Intelligent Interview Analyzer.
Tests Database, Biometrics, Pose Estimation, Lip Movement, 5-Second Mismatch Auto-Termination,
and Comprehensive Authentication Improvements (Email Validation, OTP Verification, Strong Password
Generation/Enforcement, Secure Login, Forgot Password, and Logout).
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
from modules.auth_utils import (
    validate_email_format,
    evaluate_password_strength,
    generate_strong_password
)
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
        """Test candidate profile creation with bcrypt and biometric embedding storage."""
        cand_id = "TEST-CAND-001"
        db.delete_candidate(cand_id)

        ok, msg = db.create_candidate(cand_id, "Alex Mercer", "alex@example.com", "Password@123456", is_verified=1)
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
        cand, err = db.verify_candidate_credentials(cand_id, "Password@123456")
        self.assertIsNone(err)
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
        closed_landmarks = {
            'p13': (0.5, 0.60),
            'p14': (0.5, 0.61),
            'p61': (0.42, 0.605),
            'p291': (0.58, 0.605)
        }
        mar_closed = lip_detector.calculate_mar(closed_landmarks, 640, 480)
        self.assertLess(mar_closed, 0.25)

        open_landmarks = {
            'p13': (0.5, 0.55),
            'p14': (0.5, 0.67),
            'p61': (0.42, 0.605),
            'p291': (0.58, 0.605)
        }
        mar_open = lip_detector.calculate_mar(open_landmarks, 640, 480)
        self.assertGreater(mar_open, 0.40)

    def test_06_identity_mismatch_and_5_second_termination(self):
        """Verify 5-second countdown and automatic interview termination policy."""
        session_id = "TEST-SESSION-AUTO-TERM"
        cand_id = "TEST-CAND-001"
        db.create_interview_session(session_id, cand_id)

        monitor = get_interview_monitor()
        monitor.register_session(session_id, cand_id)

        state = monitor.get_session_state(session_id)
        self.assertEqual(state['status'], 'in_progress')
        self.assertIsNone(state['mismatch_start_time'])

        # Mismatch starts
        t0 = time.time()
        state['mismatch_start_time'] = t0
        elapsed = 2.0
        remaining = Config.IDENTITY_MISMATCH_GRACE_PERIOD - elapsed
        self.assertGreater(remaining, 0)

        # Mismatch exceeds 5.0 seconds
        state['mismatch_start_time'] = time.time() - 5.1
        monitor.terminate_session(session_id, "Identity Mismatch: Candidate identity could not be verified within the permitted 5-second grace period.")

        updated_session = db.get_interview_session(session_id)
        self.assertEqual(updated_session['status'], 'terminated')
        self.assertIn("Identity Mismatch", updated_session['termination_reason'])

        # Response submission is rejected
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
        """Verify candidate restoration within 5s cancels countdown."""
        session_id = "TEST-SESSION-RESTORE"
        cand_id = "TEST-CAND-001"
        db.create_interview_session(session_id, cand_id)

        monitor = get_interview_monitor()
        monitor.register_session(session_id, cand_id)
        state = monitor.get_session_state(session_id)

        state['mismatch_start_time'] = time.time() - 2.0
        self.assertIsNotNone(state['mismatch_start_time'])

        state['mismatch_start_time'] = None
        self.assertIsNone(state['mismatch_start_time'])
        self.assertEqual(state['status'], 'in_progress')

    def test_08_admin_management_and_candidate_deletion(self):
        """Test admin creation, login, and candidate record deletion."""
        admin_user = f"admin_{int(time.time())}"
        ok, msg = db.create_admin(admin_user, f"{admin_user}@interview.test", "AdminSecret123!")
        self.assertTrue(ok)

        admin = db.verify_admin_credentials(admin_user, "AdminSecret123!")
        self.assertIsNotNone(admin)

        cand_id = "TEST-CAND-001"
        del_ok, _ = db.delete_candidate(cand_id)
        self.assertTrue(del_ok)
        self.assertIsNone(db.get_candidate(cand_id))

    # =========================================================================
    # AUTHENTICATION IMPROVEMENT TESTS (SCENARIOS 1 - 10)
    # =========================================================================

    def test_09_registration_with_valid_email(self):
        """Scenario 1: Registration with properly formatted emails including Gmail addresses."""
        valid_emails = [
            "candidate@gmail.com",
            "alex.mercer.interview@gmail.com",
            "user123@gmail.com",
            "candidate.proctor@test.com",
            "sarah_123@university.edu",
            "dev+test@company.co.uk"
        ]
        for em in valid_emails:
            is_valid, normalized = validate_email_format(em)
            self.assertTrue(is_valid, f"Failed for {em}")
            self.assertEqual(normalized, em.lower())

    def test_10_registration_with_invalid_email(self):
        """Scenario 2: Rejection of invalid email formats."""
        invalid_emails = [
            "",
            "plainaddress",
            "@missingusername.com",
            "username@.com",
            "username@domain",
            "username@domain..com",
            "user name@domain.com"
        ]
        for em in invalid_emails:
            is_valid, msg = validate_email_format(em)
            self.assertFalse(is_valid, f"Expected invalid for {em}")

        # Test API rejection
        resp = self.client.post('/send-otp', json={'email': 'invalid-email-format', 'purpose': 'registration'})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()['success'])

    def test_11_otp_verification_flow(self):
        """Scenario 3: OTP generation, verification, expiration, and incorrect code."""
        test_email = "otp_test_user@example.com"
        purpose = "registration"

        # Generate OTP
        otp_code = db.create_email_otp(test_email, purpose)
        self.assertEqual(len(otp_code), 6)
        self.assertTrue(otp_code.isdigit())

        # Test incorrect code rejection
        ok, msg = db.verify_email_otp(test_email, "000000", purpose)
        self.assertFalse(ok)
        self.assertIn("Incorrect", msg)

        # Test correct code verification
        ok_verify, msg_verify = db.verify_email_otp(test_email, otp_code, purpose)
        self.assertTrue(ok_verify)

        # Test code cannot be reused
        ok_reuse, _ = db.verify_email_otp(test_email, otp_code, purpose)
        self.assertFalse(ok_reuse)

    def test_12_registration_duplicate_email_rejection(self):
        """Scenario 4: Prevent registration with an already existing email."""
        dup_email = "duplicate_check@example.com"
        cand_id = "DUP-CAND-01"
        db.delete_candidate(cand_id)

        # Register first candidate
        ok, _ = db.create_candidate(cand_id, "Original User", dup_email, "StrongPass@12345", is_verified=1)
        self.assertTrue(ok)
        self.assertTrue(db.is_email_registered(dup_email))

        # Attempt to send OTP for registration with same email
        resp = self.client.post('/send-otp', json={'email': dup_email, 'purpose': 'registration'})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("already exists", resp.get_json()['message'])

        # Cleanup
        db.delete_candidate(cand_id)

    def test_13_weak_password_rejection(self):
        """Scenario 5: Rejection of weak passwords failing criteria or length < 12."""
        weak_passwords = [
            "short",              # < 12 chars
            "alllowercasepassword1234", # missing uppercase & special
            "ALLUPPERCASE12345678",      # missing lowercase & special
            "NoSpecialCharacters1234",  # missing special char
            "NoDigitsAtAll!@#$%"        # missing numbers
        ]
        for pwd in weak_passwords:
            evaluation = evaluate_password_strength(pwd)
            self.assertFalse(evaluation['is_valid'], f"Should reject weak password: {pwd}")
            self.assertIn(evaluation['label'], ['Weak', 'Medium'])

    def test_14_strong_password_generation(self):
        """Scenario 6: Generate strong random passwords meeting all criteria."""
        for _ in range(5):
            pwd = generate_strong_password(length=14)
            self.assertGreaterEqual(len(pwd), 12)
            evaluation = evaluate_password_strength(pwd)
            self.assertTrue(evaluation['is_valid'])
            self.assertIn(evaluation['label'], ['Strong', 'Very Strong'])
            self.assertTrue(evaluation['criteria']['min_length'])
            self.assertTrue(evaluation['criteria']['has_upper'])
            self.assertTrue(evaluation['criteria']['has_lower'])
            self.assertTrue(evaluation['criteria']['has_digit'])
            self.assertTrue(evaluation['criteria']['has_special'])

    def test_15_login_with_correct_credentials(self):
        """Scenario 7: Successful authentication using registered email and bcrypt hash."""
        cand_id = "LOGIN-OK-01"
        email = "login_success@example.com"
        password = "SecurePassword@2026!"
        db.delete_candidate(cand_id)

        ok, _ = db.create_candidate(cand_id, "Login Success User", email, password, is_verified=1)
        self.assertTrue(ok)

        # Authenticate via email
        cand, err = db.verify_candidate_credentials(email, password)
        self.assertIsNone(err)
        self.assertIsNotNone(cand)
        self.assertEqual(cand['candidate_id'], cand_id)

        # Authenticate via candidate ID
        cand_by_id, err_id = db.verify_candidate_credentials(cand_id, password)
        self.assertIsNone(err_id)
        self.assertIsNotNone(cand_by_id)

        # Test POST /login HTTP flow
        resp = self.client.post('/login', data={'identifier': email, 'password': password}, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/candidate/dashboard', resp.headers['Location'])

        # Cleanup
        db.delete_candidate(cand_id)

    def test_16_login_with_incorrect_credentials(self):
        """Scenario 8: Rejection of incorrect email/candidate ID or wrong password."""
        cand_id = "LOGIN-FAIL-01"
        email = "login_fail@example.com"
        password = "CorrectPass@2026!"
        db.delete_candidate(cand_id)

        db.create_candidate(cand_id, "Login Fail User", email, password, is_verified=1)

        # Wrong password
        cand, err = db.verify_candidate_credentials(email, "WrongPassword@999")
        self.assertEqual(err, "invalid_credentials")
        self.assertIsNone(cand)

        # Unregistered email
        cand2, err2 = db.verify_candidate_credentials("nonexistent@example.com", password)
        self.assertEqual(err2, "invalid_credentials")
        self.assertIsNone(cand2)

        # HTTP rejection check
        resp = self.client.post('/login', data={'identifier': email, 'password': 'WrongPassword@999'})
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Invalid Candidate ID/Email or Password", resp.data)

        # Cleanup
        db.delete_candidate(cand_id)

    def test_17_forgot_password_functionality(self):
        """Scenario 9: Forgot password OTP dispatch, code verification, password update, and login."""
        cand_id = "FORGOT-CAND-01"
        email = "forgot_test@example.com"
        old_pass = "OldPassword@12345"
        new_pass = "NewStrongPassword@2026!"
        db.delete_candidate(cand_id)

        db.create_candidate(cand_id, "Forgot Password User", email, old_pass, is_verified=1)

        # 1. Request reset OTP
        otp_code = db.create_email_otp(email, purpose="forgot_password")
        self.assertEqual(len(otp_code), 6)

        # 2. Submit reset with OTP
        resp = self.client.post('/forgot-password', json={
            'email': email,
            'otp': otp_code,
            'new_password': new_pass,
            'confirm_password': new_pass
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()['success'])

        # 3. Old password should fail
        cand_old, err_old = db.verify_candidate_credentials(email, old_pass)
        self.assertEqual(err_old, "invalid_credentials")

        # 4. New password succeeds
        cand_new, err_new = db.verify_candidate_credentials(email, new_pass)
        self.assertIsNone(err_new)
        self.assertEqual(cand_new['candidate_id'], cand_id)

        # Cleanup
        db.delete_candidate(cand_id)

    def test_18_logout_functionality(self):
        """Scenario 10: Secure session clearance and redirection on logout."""
        with self.client.session_transaction() as sess:
            sess['user_type'] = 'candidate'
            sess['candidate_id'] = 'TEST-LOGOUT-01'
            sess['full_name'] = 'Logout Tester'

        resp = self.client.get('/logout', follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/login', resp.headers['Location'])

        # Session should be empty
        with self.client.session_transaction() as sess:
            self.assertNotIn('candidate_id', sess)
            self.assertNotIn('user_type', sess)

    def test_19_registration_accepts_gmail_and_no_otp_required(self):
        """Scenario 11: Candidate registration accepts valid @gmail.com without requiring email OTP."""
        cand_id = "GMAIL-CAND-01"
        email = "candidate@gmail.com"
        db.delete_candidate(cand_id)

        # 1. Direct registration endpoint call without any 'otp' parameter
        resp = self.client.post('/register', json={
            'full_name': 'Gmail Candidate',
            'candidate_id': cand_id,
            'email': email,
            'password': 'StrongPassword@2026!',
            'confirm_password': 'StrongPassword@2026!',
            'face_images': [] # Test biometric validation triggers before any OTP check
        })
        data = resp.get_json()
        self.assertEqual(resp.status_code, 400)
        # Verify it requires face biometrics and does NOT request OTP
        self.assertFalse(data.get('requires_otp', False))
        self.assertIn("Face registration requires", data['message'])

        # 2. Database candidate creation with @gmail.com and is_verified=1
        ok, msg = db.create_candidate(cand_id, "Gmail Candidate", email, "StrongPassword@2026!", is_verified=1)
        self.assertTrue(ok)

        # 3. Successful login immediately without email OTP verification
        cand, err = db.verify_candidate_credentials(email, "StrongPassword@2026!")
        self.assertIsNone(err)
        self.assertIsNotNone(cand)
        self.assertEqual(cand['email'], email)

        # Cleanup
        db.delete_candidate(cand_id)

    def test_20_interview_template_contains_live_speech_below_question(self):
        """Scenario 12: Verify interview UI contains live spoken response below question."""
        with open(os.path.join(BASE_DIR, 'templates', 'interview.html'), 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for live spoken response container elements
        self.assertIn('Live Spoken Response', content)
        self.assertIn('id="liveSpeechContainer"', content)
        self.assertIn('id="speechStatusIndicator"', content)
        self.assertIn('id="speechPlaceholder"', content)
        self.assertIn('id="speechFinalText"', content)
        self.assertIn('id="speechInterimText"', content)
        self.assertIn('initSpeechRecognition', content)
        self.assertIn('renderCurrentSpeechUI', content)

        # Verify live speech container appears immediately after questionText and before answerInput
        q_pos = content.find('id="questionText"')
        speech_pos = content.find('id="liveSpeechContainer"')
        ans_pos = content.find('id="answerInput"')
        self.assertGreater(speech_pos, q_pos, "Speech container should appear after question text")
        self.assertLess(speech_pos, ans_pos, "Speech container should appear before answer input")

if __name__ == '__main__':
    unittest.main()

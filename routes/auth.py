"""
Authentication routes for Candidates and Administrators.
Implements secure email validation, OTP dispatch/verification, strong password enforcement,
Forgot Password recovery, and multi-angle webcam face biometric registration.
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from database import db
from modules.face_detection import get_face_detector
from modules.face_recognition import get_face_recognizer
from modules.auth_utils import (
    validate_email_format,
    validate_gmail_address,
    evaluate_password_strength,
    generate_strong_password,
    send_otp_email
)
from config import Config

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/send-otp', methods=['POST'])
def send_otp():
    """
    Send a 6-digit verification code to the candidate's email address.
    Enforces email format validation, duplicate checks, and a 60-second cooldown timer.
    """
    data = request.get_json() if request.is_json else request.form
    raw_email = data.get('email', '')
    purpose = data.get('purpose', 'registration')

    if purpose not in ('registration', 'forgot_password'):
        return jsonify({'success': False, 'message': 'Invalid verification purpose.'}), 400

    is_valid, email = validate_email_format(raw_email)
    if not is_valid:
        return jsonify({'success': False, 'message': email}), 400

    # Registration: prevent duplicate email
    if purpose == 'registration':
        if db.is_email_registered(email):
            return jsonify({
                'success': False, 
                'message': 'An account with this email address already exists. Please log in or use Forgot Password.'
            }), 400

    # Forgot password: email must exist
    if purpose == 'forgot_password':
        if not db.is_email_registered(email):
            return jsonify({
                'success': False, 
                'message': 'No registered candidate found with this email address.'
            }), 404

    # Cooldown timer enforcement
    can_resend, remaining_secs = db.can_resend_otp(email, purpose)
    if not can_resend:
        return jsonify({
            'success': False,
            'cooldown': True,
            'remaining_seconds': remaining_secs,
            'message': f'Please wait {remaining_secs} seconds before requesting another code.'
        }), 429

    # Generate OTP and dispatch email
    otp_code = db.create_email_otp(email, purpose)
    sent, msg = send_otp_email(email, otp_code, purpose)

    if not sent:
        return jsonify({'success': False, 'message': msg}), 500

    has_smtp = bool(Config.SMTP_HOST and Config.SMTP_USER and Config.SMTP_PASS)
    response_payload = {
        'success': True,
        'smtp_configured': has_smtp,
        'cooldown_seconds': Config.OTP_RESEND_COOLDOWN_SECONDS
    }

    if has_smtp:
        response_payload['message'] = f'A 6-digit verification code has been sent to {email}. Please check your inbox.'
    else:
        response_payload['message'] = f'Verification code generated! [Dev Mode Code: {otp_code}]'
        response_payload['dev_otp'] = otp_code

    return jsonify(response_payload)

@auth_bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    """
    Verify the 6-digit OTP code before proceeding.
    """
    data = request.get_json() if request.is_json else request.form
    raw_email = data.get('email', '')
    otp_code = data.get('otp', '')
    purpose = data.get('purpose', 'registration')

    is_valid, email = validate_email_format(raw_email)
    if not is_valid:
        return jsonify({'success': False, 'message': email}), 400

    if not otp_code or len(str(otp_code).strip()) != 6:
        return jsonify({'success': False, 'message': 'Please enter a valid 6-digit verification code.'}), 400

    ok, msg = db.verify_email_otp(email, otp_code, purpose)
    if not ok:
        return jsonify({'success': False, 'message': msg}), 400

    # Mark as verified in session
    verified_map = session.get('verified_emails', {})
    verified_map[f"{email}:{purpose}"] = True
    session['verified_emails'] = verified_map

    return jsonify({
        'success': True,
        'message': 'Email address successfully verified!'
    })

@auth_bp.route('/generate-password', methods=['GET', 'POST'])
def get_strong_password():
    """API endpoint to generate cryptographically strong passwords."""
    pwd = generate_strong_password(length=14)
    return jsonify({'password': pwd})

@auth_bp.route('/check-password', methods=['POST'])
def check_password_endpoint():
    """Evaluate password strength and return score and criteria."""
    data = request.get_json() if request.is_json else request.form
    pwd = data.get('password', '')
    eval_result = evaluate_password_strength(pwd)
    return jsonify(eval_result)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Candidate registration page and endpoint with strict Gmail validation and strong password."""
    if request.method == 'GET':
        return render_template('register.html')

    # POST - Handle registration with face embeddings
    if request.is_json:
        data = request.get_json()
        candidate_id = data.get('candidate_id', '').strip()
        full_name = data.get('full_name', '').strip()
        raw_email = data.get('email', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        face_images = data.get('face_images', [])

        # 1. Basic Fields Validation
        if not candidate_id or not full_name or not raw_email or not password:
            return jsonify({'success': False, 'message': 'All registration fields are required.'}), 400

        # 2. Strict Gmail Address Validation
        is_valid_email, email = validate_gmail_address(raw_email)
        if not is_valid_email:
            return jsonify({'success': False, 'message': email}), 400

        if db.is_email_registered(email):
            return jsonify({'success': False, 'message': 'An account with this email address already exists.'}), 400

        if db.is_candidate_id_registered(candidate_id):
            return jsonify({'success': False, 'message': 'Candidate ID already exists. Please choose a unique ID.'}), 400

        # 3. Strong Password Validation
        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Password and Confirm Password do not match.'}), 400

        pw_eval = evaluate_password_strength(password)
        if not pw_eval['is_valid']:
            return jsonify({
                'success': False,
                'message': f"Weak password rejected: {', '.join(pw_eval['errors'])}"
            }), 400

        # 4. Face Biometric Capture Validation
        if not face_images or len(face_images) < 1:
            return jsonify({
                'success': False, 
                'message': 'Face registration requires at least one clear face capture from your webcam.'
            }), 400

        detector = get_face_detector()
        recognizer = get_face_recognizer()
        valid_embeddings = []

        for item in face_images:
            angle_label = item.get('angle', 'frontal')
            img_b64 = item.get('image', '')
            frame = detector.decode_base64_frame(img_b64)
            if frame is None:
                continue

            count, faces = detector.detect_faces(frame)
            if count == 0:
                return jsonify({
                    'success': False,
                    'message': f'No face detected in the {angle_label} capture. Please ensure your face is well-lit and directly in view.'
                }), 400
            elif count > 1:
                return jsonify({
                    'success': False,
                    'message': f'Multiple faces detected in the {angle_label} capture. Only the candidate should be in frame.'
                }), 400

            emb, err = recognizer.extract_embedding(frame, faces[0]['raw'])
            if err or not emb:
                return jsonify({'success': False, 'message': f'Failed extracting face features: {err}'}), 400

            valid_embeddings.append({
                'angle': angle_label,
                'vector': emb
            })

        if not valid_embeddings:
            return jsonify({'success': False, 'message': 'Could not extract valid face embeddings from camera.'}), 400

        # 6. Save Candidate in Database (with bcrypt password hash & verified status)
        ok, msg = db.create_candidate(candidate_id, full_name, email, password, is_verified=1)
        if not ok:
            return jsonify({'success': False, 'message': msg}), 400

        # Save Biometric Embeddings
        saved = db.save_face_embeddings(candidate_id, valid_embeddings)
        if not saved:
            return jsonify({'success': False, 'message': 'Failed to save facial biometric data.'}), 500

        return jsonify({
            'success': True,
            'message': 'Registration successful! You may now log in to proceed with your interview.',
            'redirect': url_for('auth.login')
        })

    return jsonify({'success': False, 'message': 'Invalid request format.'}), 400

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Candidate login page with bcrypt validation and email verification checks."""
    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '')

        if not identifier or not password:
            flash("Please enter both your email address/Candidate ID and password.", "danger")
            return render_template('login.html')

        candidate, error_status = db.verify_candidate_credentials(identifier, password)

        if error_status == 'unverified':
            flash("Your email address has not been verified yet. Please verify your email to log in.", "warning")
            return render_template('login.html', unverified_email=candidate.get('email'))
        elif error_status == 'invalid_credentials' or not candidate:
            flash("Invalid Candidate ID/Email or Password.", "danger")
            return render_template('login.html')

        # Authentication success
        session.clear()
        session['user_type'] = 'candidate'
        session['candidate_id'] = candidate['candidate_id']
        session['full_name'] = candidate['full_name']
        session['email'] = candidate['email']
        session.permanent = True
        flash(f"Welcome back, {candidate['full_name']}!", 'success')
        return redirect(url_for('candidate.dashboard'))

    return render_template('login.html')

@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Forgot password recovery flow with email OTP and strong password update."""
    if request.method == 'GET':
        return render_template('forgot_password.html')

    data = request.get_json() if request.is_json else request.form
    raw_email = data.get('email', '').strip()
    otp_code = data.get('otp', '').strip()
    new_password = data.get('new_password', '')
    confirm_password = data.get('confirm_password', '')

    is_valid, email = validate_email_format(raw_email)
    if not is_valid:
        msg = email
        if request.is_json:
            return jsonify({'success': False, 'message': msg}), 400
        flash(msg, 'danger')
        return render_template('forgot_password.html')

    if not db.is_email_registered(email):
        msg = "No registered candidate found with this email address."
        if request.is_json:
            return jsonify({'success': False, 'message': msg}), 404
        flash(msg, 'danger')
        return render_template('forgot_password.html')

    if not otp_code:
        msg = "Please enter the 6-digit verification code."
        if request.is_json:
            return jsonify({'success': False, 'message': msg}), 400
        flash(msg, 'danger')
        return render_template('forgot_password.html')

    if new_password != confirm_password:
        msg = "New password and Confirm Password do not match."
        if request.is_json:
            return jsonify({'success': False, 'message': msg}), 400
        flash(msg, 'danger')
        return render_template('forgot_password.html')

    pw_eval = evaluate_password_strength(new_password)
    if not pw_eval['is_valid']:
        msg = f"Weak password rejected: {', '.join(pw_eval['errors'])}"
        if request.is_json:
            return jsonify({'success': False, 'message': msg}), 400
        flash(msg, 'danger')
        return render_template('forgot_password.html')

    # Verify OTP
    otp_ok, otp_msg = db.verify_email_otp(email, otp_code, purpose='forgot_password')
    if not otp_ok:
        if request.is_json:
            return jsonify({'success': False, 'message': otp_msg}), 400
        flash(otp_msg, 'danger')
        return render_template('forgot_password.html')

    # Update candidate password
    updated = db.update_candidate_password(email, new_password)
    if not updated:
        msg = "Failed to update password. Please try again."
        if request.is_json:
            return jsonify({'success': False, 'message': msg}), 500
        flash(msg, 'danger')
        return render_template('forgot_password.html')

    # Ensure email is marked verified as well
    db.mark_candidate_email_verified(email)

    success_msg = "Your password has been successfully reset! You can now log in with your new password."
    if request.is_json:
        return jsonify({
            'success': True,
            'message': success_msg,
            'redirect': url_for('auth.login')
        })

    flash(success_msg, 'success')
    return redirect(url_for('auth.login'))

@auth_bp.route('/logout')
def logout():
    """Candidate logout."""
    session.clear()
    flash("You have been securely logged out.", 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page with initial setup flow if no admin exists."""
    has_admin = db.has_any_admin()
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if action == 'register' and not has_admin:
            email = request.form.get('email', '').strip()
            ok, msg = db.create_admin(username, email, password)
            if ok:
                flash("Admin account created successfully! Please log in.", "success")
                return redirect(url_for('auth.admin_login'))
            else:
                flash(f"Error creating admin: {msg}", "danger")
        else:
            admin = db.verify_admin_credentials(username, password)
            if admin:
                session.clear()
                session['user_type'] = 'admin'
                session['admin_id'] = admin['id']
                session['admin_username'] = admin['username']
                flash(f"Logged in as Administrator ({admin['username']}).", "success")
                return redirect(url_for('admin.dashboard'))
            else:
                flash("Invalid administrator credentials.", "danger")

    return render_template('admin_login.html', has_admin=has_admin)

@auth_bp.route('/admin/logout')
def admin_logout():
    """Admin logout."""
    session.clear()
    flash("Administrator logged out.", 'info')
    return redirect(url_for('auth.admin_login'))

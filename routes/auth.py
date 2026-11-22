"""
Authentication routes for Candidates and Administrators.
Handles registration with multi-angle webcam face embedding extraction.
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from database import db
from modules.face_detection import get_face_detector
from modules.face_recognition import get_face_recognizer

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Candidate registration page and endpoint."""
    if request.method == 'GET':
        return render_template('register.html')

    # POST - Handle registration with face embeddings
    if request.is_json:
        data = request.get_json()
        candidate_id = data.get('candidate_id', '').strip()
        full_name = data.get('full_name', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        face_images = data.get('face_images', []) # List of {angle: 'frontal', image: 'data:image/...'}

        # Validation
        if not candidate_id or not full_name or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required.'}), 400

        if not face_images or len(face_images) < 1:
            return jsonify({
                'success': False, 
                'message': 'Face registration requires at least one clear face capture from your webcam.'
            }), 400

        # Process each captured face image and extract embeddings
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

        # Create candidate in database
        ok, msg = db.create_candidate(candidate_id, full_name, email, password)
        if not ok:
            return jsonify({'success': False, 'message': msg}), 400

        # Save embeddings
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
    """Candidate login page."""
    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '')

        candidate = db.verify_candidate_credentials(identifier, password)
        if candidate:
            session.clear()
            session['user_type'] = 'candidate'
            session['candidate_id'] = candidate['candidate_id']
            session['full_name'] = candidate['full_name']
            session['email'] = candidate['email']
            flash(f"Welcome back, {candidate['full_name']}!", 'success')
            return redirect(url_for('candidate.dashboard'))
        else:
            flash("Invalid Candidate ID/Email or Password.", 'danger')

    return render_template('login.html')

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

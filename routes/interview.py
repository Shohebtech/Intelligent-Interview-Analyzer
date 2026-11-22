"""
Interview Session Routes and Real-Time Frame Processing API.
Handles live camera monitoring, frame analysis, answer submissions, and termination.
"""
import uuid
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from database import db
from modules.interview_monitor import get_interview_monitor
from routes.candidate import candidate_login_required

interview_bp = Blueprint('interview', __name__, url_prefix='/interview')

@interview_bp.route('/start')
@candidate_login_required
def start():
    """Start or resume an interview session for candidate."""
    candidate_id = session['candidate_id']
    
    # Check if candidate has registered face embeddings
    embeddings = db.get_candidate_embeddings(candidate_id)
    if not embeddings:
        flash("Face biometric registration is missing. Please contact administrator.", "danger")
        return redirect(url_for('candidate.dashboard'))

    # Check for existing in-progress interview
    existing = db.get_active_interview_for_candidate(candidate_id)
    if existing:
        session_id = existing['session_id']
    else:
        session_id = f"INT-{uuid.uuid4().hex[:10].upper()}"
        db.create_interview_session(session_id, candidate_id)

    # Register in live monitor
    monitor = get_interview_monitor()
    monitor.register_session(session_id, candidate_id)

    return redirect(url_for('interview.session_view', session_id=session_id))

@interview_bp.route('/session/<session_id>')
@candidate_login_required
def session_view(session_id):
    """Render the live interview monitoring dashboard."""
    candidate_id = session['candidate_id']
    interview = db.get_interview_session(session_id)

    if not interview or interview['candidate_id'] != candidate_id:
        flash("Invalid interview session.", "danger")
        return redirect(url_for('candidate.dashboard'))

    if interview['status'] == 'terminated':
        return redirect(url_for('interview.terminated', session_id=session_id))

    if interview['status'] == 'completed':
        flash("This interview has already been completed.", "info")
        return redirect(url_for('candidate.dashboard'))

    questions = db.get_questions()
    existing_answers = db.get_candidate_answers(session_id)
    answered_qids = [a['question_id'] for a in existing_answers]

    return render_template(
        'interview.html',
        interview=interview,
        questions=questions,
        existing_answers=existing_answers,
        answered_qids=answered_qids,
        candidate_name=session.get('full_name'),
        candidate_id=candidate_id
    )

@interview_bp.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Real-time frame evaluation endpoint called by client webcam every 400-600ms.
    Input: JSON { session_id: str, frame: str (base64) }
    """
    if not request.is_json:
        return jsonify({'error': 'JSON payload required'}), 400

    data = request.get_json()
    session_id = data.get('session_id')
    frame_b64 = data.get('frame')

    if not session_id or not frame_b64:
        return jsonify({'error': 'Missing session_id or frame'}), 400

    monitor = get_interview_monitor()
    result = monitor.process_frame(session_id, frame_b64)

    # If backend determined termination, provide redirection URL
    if result.get('is_terminated'):
        result['redirect_url'] = url_for('interview.terminated', session_id=session_id)

    return jsonify(result)

@interview_bp.route('/submit_answer', methods=['POST'])
@candidate_login_required
def submit_answer():
    """Submit an answer to an interview question with backend security verification."""
    data = request.get_json() if request.is_json else request.form
    session_id = data.get('session_id')
    question_id = data.get('question_id')
    answer_text = data.get('answer_text', '').strip()

    if not session_id or not question_id or not answer_text:
        return jsonify({'success': False, 'message': 'Session, question, and answer are required.'}), 400

    interview = db.get_interview_session(session_id)
    if not interview:
        return jsonify({'success': False, 'message': 'Interview session not found.'}), 404

    # Critical security rule: Immediately stop accepting responses if terminated!
    if interview['status'] == 'terminated':
        return jsonify({
            'success': False, 
            'terminated': True,
            'message': 'Interview has been terminated. No further answers can be accepted.'
        }), 403

    if interview['status'] != 'in_progress':
        return jsonify({'success': False, 'message': 'Interview is no longer active.'}), 400

    db.save_candidate_answer(session_id, int(question_id), answer_text)
    return jsonify({'success': True, 'message': 'Answer recorded successfully.'})

@interview_bp.route('/finish', methods=['POST'])
@candidate_login_required
def finish():
    """Finalize the interview when candidate completes all questions."""
    data = request.get_json() if request.is_json else request.form
    session_id = data.get('session_id')
    
    interview = db.get_interview_session(session_id)
    if not interview:
        return jsonify({'success': False, 'message': 'Session not found'}), 404

    if interview['status'] == 'in_progress':
        db.update_interview_status(
            session_id=session_id,
            status='completed',
            final_notes='Interview successfully submitted by candidate.'
        )
        flash("Congratulations! Your interview has been successfully completed and submitted.", "success")
        return jsonify({'success': True, 'redirect_url': url_for('candidate.dashboard')})

    return jsonify({'success': False, 'message': f"Cannot finish session with status '{interview['status']}'."})

@interview_bp.route('/terminated')
def terminated():
    """Show official interview termination page."""
    session_id = request.args.get('session_id')
    interview = db.get_interview_session(session_id) if session_id else None

    # Clear candidate session to prevent further privileged actions
    session.clear()

    return render_template('terminated.html', interview=interview, session_id=session_id)

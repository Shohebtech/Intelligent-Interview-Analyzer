"""
Administrator Dashboard Routes.
Allows review of candidate registrations, interview histories, biometric status,
monitoring events, termination records, and candidate profile deletion.
"""
from flask import Blueprint, render_template, session, redirect, url_for, flash, request, jsonify
from database import db

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

def admin_login_required(view_func):
    """Enforce admin authentication."""
    def wrapper(*args, **kwargs):
        if session.get('user_type') != 'admin' or not session.get('admin_id'):
            flash("Administrator login required.", "warning")
            return redirect(url_for('auth.admin_login'))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper

@admin_bp.route('/dashboard')
@admin_login_required
def dashboard():
    """Main administrator overview."""
    candidates = db.get_all_candidates()
    interviews = db.get_all_interviews()
    return render_template(
        'admin.html',
        candidates=candidates,
        interviews=interviews,
        admin_username=session.get('admin_username')
    )

@admin_bp.route('/interview/<session_id>/events')
@admin_login_required
def interview_events(session_id):
    """Fetch monitoring event log for a specific interview session."""
    events = db.get_monitoring_events(session_id, limit=100)
    answers = db.get_candidate_answers(session_id)
    return jsonify({
        'session_id': session_id,
        'events': events,
        'answers': answers
    })

@admin_bp.route('/candidate/<candidate_id>/delete', methods=['POST'])
@admin_login_required
def delete_candidate_record(candidate_id):
    """Delete candidate and associated records from database."""
    ok, msg = db.delete_candidate(candidate_id)
    if ok:
        flash(f"Candidate {candidate_id} and all related records deleted.", "success")
    else:
        flash(f"Failed to delete candidate: {msg}", "danger")
    return redirect(url_for('admin.dashboard'))

"""
Candidate routes: Dashboard, Profile and History view.
"""
from flask import Blueprint, render_template, session, redirect, url_for, flash
from database import db

candidate_bp = Blueprint('candidate', __name__, url_prefix='/candidate')

def candidate_login_required(view_func):
    """Decorator to enforce candidate authentication."""
    def wrapper(*args, **kwargs):
        if session.get('user_type') != 'candidate' or not session.get('candidate_id'):
            flash("Please log in to access the candidate portal.", "warning")
            return redirect(url_for('auth.login'))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper

@candidate_bp.route('/dashboard')
@candidate_login_required
def dashboard():
    """Candidate personal dashboard."""
    candidate_id = session['candidate_id']
    profile = db.get_candidate(candidate_id)
    embeddings = db.get_candidate_embeddings(candidate_id)
    
    # Retrieve past interviews
    conn = db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM interviews 
    WHERE candidate_id = ? 
    ORDER BY start_time DESC
    """, (candidate_id,))
    interviews = [dict(r) for r in cursor.fetchall()]
    conn.close()

    active_interview = db.get_active_interview_for_candidate(candidate_id)

    return render_template(
        'dashboard.html',
        profile=profile,
        embedding_count=len(embeddings),
        interviews=interviews,
        active_interview=active_interview
    )

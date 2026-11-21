"""
Database management for Intelligent Interview Analyzer.
SQLite-based storage with robust connection handling and schema initialization.
"""
import sqlite3
import os
import json
import numpy as np
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config

def get_db_connection():
    """Get a SQLite database connection with row factory configured."""
    conn = sqlite3.connect(Config.DATABASE, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """Initialize database tables and default interview questions."""
    os.makedirs(os.path.dirname(Config.DATABASE), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Candidates table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. Face Embeddings table (stores multiple angle embeddings per candidate)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        angle_label TEXT DEFAULT 'frontal',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE
    )
    """)

    # 3. Interviews table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS interviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE NOT NULL,
        candidate_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'in_progress', -- 'in_progress', 'completed', 'terminated'
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP,
        warning_count INTEGER DEFAULT 0,
        termination_reason TEXT,
        final_notes TEXT,
        FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id) ON DELETE CASCADE
    )
    """)

    # 4. Monitoring Events table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS monitoring_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        candidate_id TEXT NOT NULL,
        event_type TEXT NOT NULL, -- 'VERIFIED', 'MISMATCH', 'NO_FACE', 'MULTIPLE_FACES', 'HEAD_YAW', 'HEAD_PITCH', 'LIP_SPEAKING', 'WARNING', 'TERMINATED'
        severity TEXT NOT NULL DEFAULT 'INFO', -- 'INFO', 'WARNING', 'CRITICAL'
        details TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES interviews(session_id) ON DELETE CASCADE
    )
    """)

    # 5. Admins table (no default hardcoded credentials - user creates them)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 6. Standard Interview Questions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS interview_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_order INTEGER NOT NULL,
        question_text TEXT NOT NULL,
        category TEXT DEFAULT 'Technical & Behavioral',
        expected_duration_sec INTEGER DEFAULT 120
    )
    """)

    # 7. Candidate Answers during interview
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidate_answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        question_id INTEGER NOT NULL,
        answer_text TEXT NOT NULL,
        submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES interviews(session_id) ON DELETE CASCADE,
        FOREIGN KEY (question_id) REFERENCES interview_questions(id) ON DELETE CASCADE
    )
    """)

    # Seed standard questions if table is empty
    cursor.execute("SELECT COUNT(*) as cnt FROM interview_questions")
    if cursor.fetchone()['cnt'] == 0:
        seed_questions = [
            (1, "Please introduce yourself, your technical background, and your key areas of expertise.", "Introduction", 120),
            (2, "Describe a challenging technical problem you recently solved. What was your systematic approach?", "Technical Problem Solving", 180),
            (3, "How do you ensure high reliability and security when architecting distributed web systems?", "Architecture & Security", 180),
            (4, "Explain a scenario where you faced conflicting requirements or deadlines, and how you navigated it.", "Behavioral & Collaboration", 120),
            (5, "Where do you see your technical focus evolving over the next two years, and what drives your curiosity?", "Future Vision", 120)
        ]
        cursor.executemany("""
        INSERT INTO interview_questions (question_order, question_text, category, expected_duration_sec)
        VALUES (?, ?, ?, ?)
        """, seed_questions)

    conn.commit()
    conn.close()

# Candidate helper functions
def create_candidate(candidate_id, full_name, email, password):
    """Register a new candidate."""
    conn = get_db_connection()
    cursor = conn.cursor()
    password_hash = generate_password_hash(password)
    try:
        cursor.execute("""
        INSERT INTO candidates (candidate_id, full_name, email, password_hash)
        VALUES (?, ?, ?, ?)
        """, (candidate_id.strip(), full_name.strip(), email.strip().lower(), password_hash))
        conn.commit()
        return True, "Candidate registered successfully"
    except sqlite3.IntegrityError as e:
        if "candidate_id" in str(e):
            return False, "Candidate ID already exists."
        elif "email" in str(e):
            return False, "Email address already registered."
        return False, f"Registration failed: {str(e)}"
    finally:
        conn.close()

def save_face_embeddings(candidate_id, embeddings_list):
    """
    Save multiple face embeddings for candidate.
    embeddings_list: list of dicts [{'angle': 'frontal', 'vector': [float, ...]}]
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        for item in embeddings_list:
            angle = item.get('angle', 'frontal')
            vector = item.get('vector')
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            cursor.execute("""
            INSERT INTO face_embeddings (candidate_id, embedding_json, angle_label)
            VALUES (?, ?, ?)
            """, (candidate_id, json.dumps(vector), angle))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error saving embeddings: {e}")
        return False
    finally:
        conn.close()

def get_candidate_embeddings(candidate_id):
    """Retrieve all stored embeddings for a candidate as list of numpy arrays."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT embedding_json, angle_label FROM face_embeddings WHERE candidate_id = ?
    """, (candidate_id,))
    rows = cursor.fetchall()
    conn.close()

    embeddings = []
    for r in rows:
        try:
            vec = json.loads(r['embedding_json'])
            embeddings.append(np.array(vec, dtype=np.float32))
        except Exception as e:
            print(f"Error parsing embedding: {e}")
    return embeddings

def verify_candidate_credentials(identifier, password):
    """Authenticate candidate by candidate_id or email."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM candidates WHERE candidate_id = ? OR email = ?
    """, (identifier.strip(), identifier.strip().lower()))
    candidate = cursor.fetchone()
    conn.close()

    if candidate and check_password_hash(candidate['password_hash'], password):
        return dict(candidate)
    return None

def get_candidate(candidate_id):
    """Retrieve candidate profile."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# Interview session helpers
def create_interview_session(session_id, candidate_id):
    """Create a new active interview session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO interviews (session_id, candidate_id, status, start_time)
    VALUES (?, ?, 'in_progress', CURRENT_TIMESTAMP)
    """, (session_id, candidate_id))
    conn.commit()
    conn.close()

def get_interview_session(session_id):
    """Get interview session details."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT i.*, c.full_name, c.email 
    FROM interviews i
    JOIN candidates c ON i.candidate_id = c.candidate_id
    WHERE i.session_id = ?
    """, (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_active_interview_for_candidate(candidate_id):
    """Check if candidate already has an in-progress interview."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM interviews 
    WHERE candidate_id = ? AND status = 'in_progress'
    ORDER BY start_time DESC LIMIT 1
    """, (candidate_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_interview_status(session_id, status, termination_reason=None, final_notes=None):
    """Update interview status (completed, terminated)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE interviews 
    SET status = ?, end_time = CURRENT_TIMESTAMP, termination_reason = ?, final_notes = ?
    WHERE session_id = ?
    """, (status, termination_reason, final_notes, session_id))
    conn.commit()
    conn.close()

def increment_interview_warning(session_id):
    """Increment warning count for session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE interviews SET warning_count = warning_count + 1 WHERE session_id = ?
    """, (session_id,))
    conn.commit()
    cursor.execute("SELECT warning_count FROM interviews WHERE session_id = ?", (session_id,))
    count = cursor.fetchone()['warning_count']
    conn.close()
    return count

def log_monitoring_event(session_id, candidate_id, event_type, severity='INFO', details=None):
    """Log an event during interview monitoring."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO monitoring_events (session_id, candidate_id, event_type, severity, details)
    VALUES (?, ?, ?, ?, ?)
    """, (session_id, candidate_id, event_type, severity, details))
    conn.commit()
    conn.close()

def get_monitoring_events(session_id, limit=50):
    """Retrieve recent monitoring events for session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM monitoring_events 
    WHERE session_id = ? 
    ORDER BY timestamp DESC LIMIT ?
    """, (session_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Admin helpers
def create_admin(username, email, password):
    """Create an administrator account."""
    conn = get_db_connection()
    cursor = conn.cursor()
    password_hash = generate_password_hash(password)
    try:
        cursor.execute("""
        INSERT INTO admins (username, email, password_hash)
        VALUES (?, ?, ?)
        """, (username.strip(), email.strip().lower(), password_hash))
        conn.commit()
        return True, "Admin created successfully"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Admin username already exists."
        elif "email" in str(e):
            return False, "Email already registered."
        return False, str(e)
    finally:
        conn.close()

def verify_admin_credentials(username_or_email, password):
    """Verify administrator login credentials."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM admins WHERE username = ? OR email = ?
    """, (username_or_email.strip(), username_or_email.strip().lower()))
    admin = cursor.fetchone()
    conn.close()
    if admin and check_password_hash(admin['password_hash'], password):
        return dict(admin)
    return None

def has_any_admin():
    """Check if any admin account exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM admins")
    cnt = cursor.fetchone()['cnt']
    conn.close()
    return cnt > 0

def get_all_candidates():
    """Get all registered candidates with interview stats."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT c.*, 
           COUNT(DISTINCT fe.id) as embedding_count,
           COUNT(DISTINCT i.id) as interview_count,
           MAX(i.start_time) as last_interview_time
    FROM candidates c
    LEFT JOIN face_embeddings fe ON c.candidate_id = fe.candidate_id
    LEFT JOIN interviews i ON c.candidate_id = i.candidate_id
    GROUP BY c.candidate_id
    ORDER BY c.created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_interviews():
    """Get all interview records with candidate info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT i.*, c.full_name, c.email
    FROM interviews i
    JOIN candidates c ON i.candidate_id = c.candidate_id
    ORDER BY i.start_time DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_candidate(candidate_id):
    """Delete a candidate and cascade delete embeddings, interviews, and events."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Delete related answers and events
        cursor.execute("SELECT session_id FROM interviews WHERE candidate_id = ?", (candidate_id,))
        sessions = [r['session_id'] for r in cursor.fetchall()]
        for s in sessions:
            cursor.execute("DELETE FROM candidate_answers WHERE session_id = ?", (s,))
            cursor.execute("DELETE FROM monitoring_events WHERE session_id = ?", (s,))
        cursor.execute("DELETE FROM interviews WHERE candidate_id = ?", (candidate_id,))
        cursor.execute("DELETE FROM face_embeddings WHERE candidate_id = ?", (candidate_id,))
        cursor.execute("DELETE FROM candidates WHERE candidate_id = ?", (candidate_id,))
        conn.commit()
        return True, "Candidate deleted successfully."
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        conn.close()

def get_questions():
    """Fetch ordered list of interview questions."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interview_questions ORDER BY question_order ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_candidate_answer(session_id, question_id, answer_text):
    """Save an answer submitted by candidate."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO candidate_answers (session_id, question_id, answer_text)
    VALUES (?, ?, ?)
    """, (session_id, question_id, answer_text.strip()))
    conn.commit()
    conn.close()

def get_candidate_answers(session_id):
    """Get all submitted answers for an interview session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT ca.*, q.question_text, q.category, q.question_order
    FROM candidate_answers ca
    JOIN interview_questions q ON ca.question_id = q.id
    WHERE ca.session_id = ?
    ORDER BY q.question_order ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

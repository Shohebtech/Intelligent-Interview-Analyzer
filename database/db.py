"""
Database management for Intelligent Interview Analyzer.
SQLite-based storage with robust connection handling, bcrypt password hashing,
schema initialization, and email OTP management.
"""
import sqlite3
import os
import json
import time
import secrets
import bcrypt
import numpy as np
from datetime import datetime
from config import Config

def get_db_connection():
    """Get a SQLite database connection with row factory configured."""
    conn = sqlite3.connect(Config.DATABASE, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def hash_password(password: str) -> str:
    """Hash password using bcrypt with standard work factor."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def check_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash, with fallback to legacy hashes."""
    if not plain_password or not hashed_password:
        return False
    try:
        # Check if bcrypt hash format ($2a$, $2b$, $2y$)
        if hashed_password.startswith(('$2b$', '$2a$', '$2y$')):
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        # Fallback for werkzeug hashes if previously stored
        from werkzeug.security import check_password_hash
        return check_password_hash(hashed_password, plain_password)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

def init_db():
    """Initialize database tables, schema migrations, and default interview questions."""
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
        is_verified INTEGER DEFAULT 0,
        verified_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Schema migration: check if is_verified column exists
    cursor.execute("PRAGMA table_info(candidates)")
    cols = [r['name'] for r in cursor.fetchall()]
    if 'is_verified' not in cols:
        try:
            cursor.execute("ALTER TABLE candidates ADD COLUMN is_verified INTEGER DEFAULT 0")
            cursor.execute("ALTER TABLE candidates ADD COLUMN verified_at TIMESTAMP")
            # Set existing registered candidates to verified to prevent breaking previous sessions
            cursor.execute("UPDATE candidates SET is_verified = 1 WHERE is_verified = 0")
            conn.commit()
        except Exception as e:
            print(f"Migration notice: {e}")

    # 2. Email OTPs table for verification & password reset
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS email_otps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        otp_code TEXT NOT NULL,
        purpose TEXT NOT NULL, -- 'registration', 'forgot_password'
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_epoch REAL NOT NULL,
        expires_epoch REAL NOT NULL,
        is_used INTEGER DEFAULT 0
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_email_otps ON email_otps(email, purpose, is_used)")

    # 3. Face Embeddings table (stores multiple angle embeddings per candidate)
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

    # 4. Interviews table
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

    # 5. Monitoring Events table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS monitoring_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        candidate_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        severity TEXT NOT NULL DEFAULT 'INFO',
        details TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES interviews(session_id) ON DELETE CASCADE
    )
    """)

    # 6. Admins table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 7. Standard Interview Questions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS interview_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question_order INTEGER NOT NULL,
        question_text TEXT NOT NULL,
        category TEXT DEFAULT 'Technical & Behavioral',
        expected_duration_sec INTEGER DEFAULT 120
    )
    """)

    # 8. Candidate Answers during interview
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

# OTP Helpers
def can_resend_otp(email: str, purpose: str = "registration"):
    """
    Check if a new OTP can be requested (enforces cooldown timer).
    Returns (can_resend: bool, seconds_remaining: int)
    """
    email_clean = email.strip().lower()
    cooldown = Config.OTP_RESEND_COOLDOWN_SECONDS
    now = time.time()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT created_epoch FROM email_otps 
    WHERE email = ? AND purpose = ?
    ORDER BY created_epoch DESC LIMIT 1
    """, (email_clean, purpose))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return True, 0
        
    elapsed = now - row['created_epoch']
    if elapsed < cooldown:
        return False, int(cooldown - elapsed)
        
    return True, 0

def create_email_otp(email: str, purpose: str = "registration") -> str:
    """
    Generate and save a 6-digit numeric OTP in database with expiration.
    Returns the generated OTP string.
    """
    email_clean = email.strip().lower()
    # 6-digit numeric OTP
    otp_code = str(secrets.randbelow(900000) + 100000)
    now = time.time()
    expires_at = now + (Config.OTP_EXPIRY_MINUTES * 60)

    conn = get_db_connection()
    cursor = conn.cursor()
    # Invalidate previous unused OTPs for this email & purpose
    cursor.execute("""
    UPDATE email_otps SET is_used = 1 
    WHERE email = ? AND purpose = ? AND is_used = 0
    """, (email_clean, purpose))

    cursor.execute("""
    INSERT INTO email_otps (email, otp_code, purpose, created_epoch, expires_epoch, is_used)
    VALUES (?, ?, ?, ?, ?, 0)
    """, (email_clean, otp_code, purpose, now, expires_at))
    conn.commit()
    conn.close()
    return otp_code

def verify_email_otp(email: str, otp_code: str, purpose: str = "registration"):
    """
    Validate provided OTP against active, unexpired database records.
    Returns (success: bool, message: str)
    """
    email_clean = email.strip().lower()
    otp_clean = str(otp_code).strip()
    now = time.time()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM email_otps 
    WHERE email = ? AND purpose = ? AND is_used = 0
    ORDER BY created_epoch DESC LIMIT 1
    """, (email_clean, purpose))
    record = cursor.fetchone()

    if not record:
        conn.close()
        return False, "No active verification code found for this email. Please request a new code."

    if record['expires_epoch'] < now:
        conn.close()
        return False, "Verification code has expired. Please request a new code."

    if record['otp_code'] != otp_clean:
        conn.close()
        return False, "Incorrect verification code. Please check and try again."

    # Mark as used
    cursor.execute("UPDATE email_otps SET is_used = 1 WHERE id = ?", (record['id'],))
    conn.commit()
    conn.close()
    return True, "Verification successful."

# Candidate helper functions
def is_email_registered(email: str) -> bool:
    """Check if an email address is already registered."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM candidates WHERE email = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return bool(row)

def is_candidate_id_registered(candidate_id: str) -> bool:
    """Check if a candidate ID already exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM candidates WHERE candidate_id = ?", (candidate_id.strip(),))
    row = cursor.fetchone()
    conn.close()
    return bool(row)

def create_candidate(candidate_id, full_name, email, password, is_verified=1):
    """Register a new candidate with bcrypt password hash."""
    conn = get_db_connection()
    cursor = conn.cursor()
    pw_hash = hash_password(password)
    email_clean = email.strip().lower()
    cand_id_clean = candidate_id.strip()

    try:
        cursor.execute("""
        INSERT INTO candidates (candidate_id, full_name, email, password_hash, is_verified, verified_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (cand_id_clean, full_name.strip(), email_clean, pw_hash, 1 if is_verified else 0))
        conn.commit()
        return True, "Candidate registered successfully."
    except sqlite3.IntegrityError as e:
        if "candidate_id" in str(e):
            return False, "Candidate ID already exists."
        elif "email" in str(e):
            return False, "An account with this email address already exists."
        return False, f"Registration failed: {str(e)}"
    finally:
        conn.close()

def update_candidate_password(email: str, new_password: str):
    """Update candidate password with new bcrypt hash."""
    email_clean = email.strip().lower()
    pw_hash = hash_password(new_password)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE candidates SET password_hash = ? WHERE email = ?
    """, (pw_hash, email_clean))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def mark_candidate_email_verified(email: str):
    """Set email verification flag to true."""
    email_clean = email.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE candidates SET is_verified = 1, verified_at = CURRENT_TIMESTAMP WHERE email = ?
    """, (email_clean,))
    conn.commit()
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
    """
    Authenticate candidate by candidate_id or email using bcrypt verification.
    Returns (candidate_dict, error_status) where error_status can be:
    - None (success)
    - 'unverified' (valid credentials but email not verified)
    - 'invalid_credentials' (wrong user/password)
    """
    ident = identifier.strip()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM candidates WHERE candidate_id = ? OR email = ?
    """, (ident, ident.lower()))
    candidate = cursor.fetchone()
    conn.close()

    if not candidate:
        return None, "invalid_credentials"

    cand_dict = dict(candidate)
    if not check_password(password, cand_dict['password_hash']):
        return None, "invalid_credentials"

    # Check email verification status
    if not cand_dict.get('is_verified', 1):
        return cand_dict, "unverified"

    return cand_dict, None

def get_candidate(candidate_id):
    """Retrieve candidate profile."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_candidate_by_email(email):
    """Retrieve candidate profile by email."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM candidates WHERE email = ?", (email.strip().lower(),))
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
    """Create an administrator account with bcrypt."""
    conn = get_db_connection()
    cursor = conn.cursor()
    password_hash = hash_password(password)
    try:
        cursor.execute("""
        INSERT INTO admins (username, email, password_hash)
        VALUES (?, ?, ?)
        """, (username.strip(), email.strip().lower(), password_hash))
        conn.commit()
        return True, "Admin created successfully."
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
    if admin and check_password(password, admin['password_hash']):
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

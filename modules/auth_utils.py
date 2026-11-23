"""
Authentication utilities for Intelligent Interview Analyzer.
Handles email format validation, secure OTP generation, strong password generation,
password strength evaluation, and email dispatch via SMTP.
"""
import re
import secrets
import string
import smtplib
import os
import json
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import Config

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')

def validate_email_format(email):
    """
    Validate email address format.
    Returns (is_valid: bool, normalized_email_or_error: str)
    """
    if not email or not isinstance(email, str):
        return False, "Email address is required."
    
    email = email.strip()
    if len(email) > 254:
        return False, "Email address is too long."
        
    if not EMAIL_REGEX.match(email):
        return False, "Please enter a valid email address (e.g., user@example.com)."
        
    if '..' in email:
        return False, "Email address cannot contain consecutive periods."

    parts = email.split('@')
    if len(parts) != 2:
        return False, "Invalid email address format."
        
    domain = parts[1]
    if '.' not in domain or domain.startswith('.') or domain.endswith('.'):
        return False, "Invalid email domain."
        
    return True, email.lower()

def generate_strong_password(length=14):
    """
    Generate a cryptographically secure random strong password containing:
    - Uppercase letters (A-Z)
    - Lowercase letters (a-z)
    - Numbers (0-9)
    - Special characters (@, #, $, %, &, !, *, etc.)
    Minimum length: 12 characters.
    """
    length = max(12, length)
    
    uppers = string.ascii_uppercase
    lowers = string.ascii_lowercase
    digits = string.digits
    specials = "@#$%&*!_-+=~"
    all_chars = uppers + lowers + digits + specials

    password_chars = [
        secrets.choice(uppers),
        secrets.choice(lowers),
        secrets.choice(digits),
        secrets.choice(specials),
        secrets.choice(uppers),
        secrets.choice(lowers),
        secrets.choice(digits),
        secrets.choice(specials)
    ]
    
    for _ in range(length - len(password_chars)):
        password_chars.append(secrets.choice(all_chars))
        
    shuffled = []
    while password_chars:
        idx = secrets.randbelow(len(password_chars))
        shuffled.append(password_chars.pop(idx))
        
    return "".join(shuffled)

def evaluate_password_strength(password):
    """
    Evaluate password strength.
    Requires all 5 criteria to be considered valid and Strong.
    """
    if not password:
        return {
            'score': 0,
            'label': 'Weak',
            'is_valid': False,
            'errors': ['Password is required.'],
            'criteria': {
                'min_length': False,
                'has_upper': False,
                'has_lower': False,
                'has_digit': False,
                'has_special': False
            }
        }

    criteria = {
        'min_length': len(password) >= 12,
        'has_upper': bool(re.search(r'[A-Z]', password)),
        'has_lower': bool(re.search(r'[a-z]', password)),
        'has_digit': bool(re.search(r'[0-9]', password)),
        'has_special': bool(re.search(r'[@#$%^&*!_+\-=\[\]{}|;:,.<>?~]', password))
    }

    errors = []
    if not criteria['min_length']:
        errors.append("At least 12 characters required.")
    if not criteria['has_upper']:
        errors.append("At least one uppercase letter (A-Z) required.")
    if not criteria['has_lower']:
        errors.append("At least one lowercase letter (a-z) required.")
    if not criteria['has_digit']:
        errors.append("At least one number (0-9) required.")
    if not criteria['has_special']:
        errors.append("At least one special character (@, #, $, %, &, etc.) required.")

    passed_count = sum(criteria.values())
    is_valid = (passed_count == 5)

    if not criteria['min_length'] or passed_count <= 2:
        score = 1
        label = 'Weak'
    elif passed_count in (3, 4):
        score = 2
        label = 'Medium'
    else: # all 5 passed
        if len(password) >= 14:
            score = 4
            label = 'Very Strong'
        else:
            score = 3
            label = 'Strong'

    return {
        'score': score,
        'label': label,
        'is_valid': is_valid,
        'errors': errors,
        'criteria': criteria
    }

def send_otp_email(recipient_email, otp_code, purpose="registration"):
    """
    Dispatch OTP verification email.
    If SMTP server credentials are provided in Config / environment, sends real email via SMTP.
    In development or fallback mode, logs OTP to instance/otp_dev_log.json and console.
    """
    subject = "Your Verification Code - Intelligent Interview Analyzer"
    if purpose == "forgot_password":
        subject = "Password Reset Code - Intelligent Interview Analyzer"
        
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0b132b; color: #f8fafc; padding: 20px; }}
        .card {{ background-color: #141e3c; border: 1px solid #23315a; border-radius: 12px; padding: 30px; max-width: 500px; margin: 0 auto; box-shadow: 0 10px 25px rgba(0,0,0,0.4); }}
        .header {{ font-size: 20px; font-weight: bold; color: #60a5fa; margin-bottom: 20px; text-align: center; }}
        .otp-box {{ background-color: #070d1e; border: 2px dashed #7c3aed; border-radius: 8px; padding: 16px; text-align: center; margin: 25px 0; }}
        .otp-code {{ font-size: 36px; font-weight: 800; letter-spacing: 8px; color: #38bdf8; font-family: monospace; }}
        .info {{ font-size: 14px; color: #94a3b8; line-height: 1.6; }}
        .footer {{ margin-top: 25px; font-size: 12px; color: #64748b; text-align: center; border-top: 1px solid #23315a; padding-top: 15px; }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="header">Intelligent Interview Analyzer</div>
        <p class="info">Hello,</p>
        <p class="info">Use the verification code below to complete your { 'registration' if purpose == 'registration' else 'password reset' }:</p>
        <div class="otp-box">
          <div class="otp-code">{otp_code}</div>
        </div>
        <p class="info">This code is valid for <strong>10 minutes</strong>. For security, never share this code with anyone.</p>
        <div class="footer">
          If you did not request this verification code, please ignore this email.
        </div>
      </div>
    </body>
    </html>
    """

    try:
        dev_log_path = os.path.join(Config.INSTANCE_DIR, 'otp_dev_log.json')
        dev_entry = {
            'email': recipient_email,
            'otp': otp_code,
            'purpose': purpose,
            'timestamp': time.time(),
            'formatted_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        logs = []
        if os.path.exists(dev_log_path):
            with open(dev_log_path, 'r') as f:
                try:
                    logs = json.load(f)
                except Exception:
                    logs = []
        logs.append(dev_entry)
        logs = logs[-50:]
        with open(dev_log_path, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Notice writing dev OTP log: {e}")

    if Config.SMTP_HOST and Config.SMTP_USER and Config.SMTP_PASS:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = Config.MAIL_DEFAULT_SENDER or Config.SMTP_USER
            msg["To"] = recipient_email
            
            part = MIMEText(html_content, "html")
            msg.attach(part)

            server = smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT, timeout=10)
            if Config.SMTP_USE_TLS:
                server.starttls()
            server.login(Config.SMTP_USER, Config.SMTP_PASS)
            server.sendmail(msg["From"], [recipient_email], msg.as_string())
            server.quit()
            return True, "Verification code sent to your email."
        except Exception as err:
            print(f"SMTP send error: {err}")
            return False, f"Failed to send email: {str(err)}"

    print(f"=== [LOCAL DEV OTP DELIVERY] === To: {recipient_email} | Code: {otp_code} | Purpose: {purpose} ===")
    return True, "Verification code generated and sent (check your email/console)."

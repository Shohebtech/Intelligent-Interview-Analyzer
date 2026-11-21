"""
Intelligent Interview Analyzer - Main Flask Server.
Conducts real-time interview integrity monitoring using OpenCV and MediaPipe.
"""
import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Optimize Matplotlib cache directory
os.environ['MPLCONFIGDIR'] = os.path.join(BASE_DIR, 'instance', 'matplotlib_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

from flask import Flask, render_template
from config import Config
from database import db
from routes.auth import auth_bp
from routes.candidate import candidate_bp
from routes.interview import interview_bp
from routes.admin import admin_bp

def create_app():
    """Application factory for Intelligent Interview Analyzer."""
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)

    # Initialize Database Schema
    db.init_db()

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(candidate_bp)
    app.register_blueprint(interview_bp)
    app.register_blueprint(admin_bp)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.errorhandler(404)
    def not_found(e):
        return render_template('base.html'), 404

    @app.errorhandler(500)
    def internal_error(e):
        return render_template('base.html'), 500

    return app

if __name__ == '__main__':
    app = create_app()
    print("=========================================================")
    print("  Intelligent Interview Analyzer Server Starting")
    print(f"  Access locally at: http://localhost:{Config.PORT}")
    print("=========================================================")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)

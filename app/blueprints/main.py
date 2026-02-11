from flask import Blueprint, jsonify, render_template

bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    return render_template('logs.html')


@bp.route('/info')
def info():
    return render_template('api.html')


@bp.route('/detect-viewer')
def detect_viewer():
    return render_template('detect.html')


@bp.route('/health')
def health():
    from app import get_detection_service
    from flask import current_app
    
    detection_service = get_detection_service()
    simulation_mode = current_app.config['SIMULATION_MODE']
    
    model_loaded = False
    if not simulation_mode:
        model_loaded = detection_service.is_model_loaded()
    
    return jsonify({
        "status": "healthy",
        "simulation_mode": simulation_mode,
        "model_loaded": model_loaded,
        "model_name": detection_service.get_current_model() if not simulation_mode else None
    })

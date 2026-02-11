from flask import Blueprint, jsonify

bp = Blueprint('capture', __name__)


@bp.route('/start_capture', methods=['POST'])
def start_capture():
    from app import get_capture_service, get_logger
    from app.services.logger_service import LogLevel
    
    capture_service = get_capture_service()
    logger = get_logger()
    
    capture_service.start_capture()
    logger.log(LogLevel.INFO, "Capture started via API", "start_capture_endpoint")
    return jsonify({"status": "Capture started"})


@bp.route('/stop_capture', methods=['POST'])
def stop_capture():
    from app import get_capture_service, get_logger
    from app.services.logger_service import LogLevel
    
    capture_service = get_capture_service()
    logger = get_logger()
    
    capture_service.stop_capture()
    logger.log(LogLevel.INFO, "Capture stopped via API", "stop_capture_endpoint")
    return jsonify({"status": "Capture stopped"})

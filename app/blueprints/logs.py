from flask import Blueprint, jsonify, request
from app.services.logger_service import LogLevel

bp = Blueprint('logs', __name__)


@bp.route('/logs', methods=['POST'])
def post_log():
    from app import get_logger
    
    logger = get_logger()
    
    try:
        data = request.json
        level_str = data.get("level", "INFO").upper()
        message = data.get("message", "")
        context = data.get("context", "")
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        try:
            log_level = LogLevel[level_str]
        except KeyError:
            return jsonify({"error": f"Invalid log level: {level_str}. Must be one of: ERROR, WARNING, INFO, DETECTION, DECISION"}), 400
        
        logger.log(log_level, message, context)
        return jsonify({"status": "Log recorded", "level": level_str, "message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/logs', methods=['GET'])
def get_logs():
    from app import get_logger
    
    logger = get_logger()
    all_logs = logger.get_all_logs()
    return jsonify({"logs": all_logs})


@bp.route('/logs/<level>', methods=['GET'])
def get_logs_by_level(level):
    from app import get_logger
    
    logger = get_logger()
    
    try:
        log_level = LogLevel[level.upper()]
        logs = logger.get_logs_by_level(log_level)
        return jsonify({
            "level": level.upper(),
            "logs": logs,
            "count": len(logs)
        })
    except KeyError:
        return jsonify({"error": f"Unknown log level: {level}"}), 400


@bp.route('/logs/clear', methods=['POST'])
def clear_logs():
    from app import get_logger
    
    logger = get_logger()
    logger.clear_logs()
    return jsonify({"status": "Logs cleared"})

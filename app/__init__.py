import warnings
from flask import Flask
from app.config import Config, HAILO_MODEL_PATH
from app.services.capture_service import CaptureService
from app.services.detection_service import DetectionService
from app.services.logger_service import Logger

warnings.filterwarnings("ignore", category=RuntimeWarning)

capture_service = None
detection_service = None
logger = None


def create_app(config_class=Config):
    global capture_service, detection_service, logger
    
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    logger = Logger()
    
    # Initialize detection service based on config
    detection_service = DetectionService(
            simulation_mode=app.config['SIMULATION_MODE'],
            model_name=app.config['DEFAULT_MODEL']
        )
    
    capture_service = CaptureService(
        simulation_mode=app.config['SIMULATION_MODE'],
        detection_service=detection_service,
        logger=logger
    )
    
    from app.blueprints.main import bp as main_bp
    from app.blueprints.detection import bp as detection_bp
    from app.blueprints.logs import bp as logs_bp
    from app.blueprints.capture import bp as capture_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(logs_bp)
    app.register_blueprint(capture_bp)
    
    return app


def get_capture_service():
    return capture_service


def get_detection_service():
    return detection_service


def get_logger():
    return logger


def set_detection_service(service_type, **kwargs):
    """Switch detection service at runtime"""
    global detection_service
    if service_type.lower() == 'yolo':
        from app.config import YOLO_MODEL_PATHS
        model_name = kwargs.get('model', 'yolo11n')
        simulation_mode = kwargs.get('simulation_mode', False)
        if model_name not in YOLO_MODEL_PATHS:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(YOLO_MODEL_PATHS.keys())}")
        detection_service = DetectionService(
            simulation_mode=simulation_mode,
            model_name=model_name
        )
        return {"status": f"Switched to YOLO ({model_name})", "service": "yolo", "model": model_name}
    else:
        raise ValueError(f"Unknown service type: {service_type}. Use 'yolo'")
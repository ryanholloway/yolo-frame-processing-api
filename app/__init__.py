import warnings
from flask import Flask
from app.config import Config
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

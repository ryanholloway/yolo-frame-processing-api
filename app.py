from flask import Flask, jsonify, request, Response
import threading
import time
import cv2
import numpy as np
import warnings
from io import BytesIO
warnings.filterwarnings("ignore", category=RuntimeWarning)
from ultralytics import YOLO
from config import YOLO_MODEL_PATHS, CUSTOM_CLASS_NAMES, colors, create_fake_image, create_message_frame, IMAGE_WIDTH, IMAGE_HEIGHT
from camera import Camera
from logger import Logger, LogLevel

app = Flask(__name__)

# Initialize global logger
logger = Logger()

SIMULATION_MODE = False

camera = Camera(simulation_mode=SIMULATION_MODE)

if not SIMULATION_MODE:
    model_name = "yolo11n"
    try:
        model = YOLO(YOLO_MODEL_PATHS[model_name])
        print(f"Model loaded successfully from {YOLO_MODEL_PATHS[model_name]}")
    except Exception as e:
        print(f"Error loading model: {e}")

latest_frame = None
latest_detections = []
data_lock = threading.Lock()
is_capturing = False
capture_thread_instance = None

def start_capture():
    global is_capturing, capture_thread_instance
    if not is_capturing:
        is_capturing = True
        capture_thread_instance = threading.Thread(target=capture_thread, daemon=True)
        capture_thread_instance.start()

def stop_capture():
    global is_capturing
    is_capturing = False

def capture_thread():
    global latest_frame, latest_detections
    while is_capturing:
        frame = camera.capture()
        if SIMULATION_MODE:
            import random
            num_detections = random.randint(1, 5)
            fake_detections = []
            for _ in range(num_detections):
                class_name = random.choice(CUSTOM_CLASS_NAMES)
                confidence = round(random.uniform(0.5, 1.0), 2)
                fake_detections.append({
                    "class_name": class_name,
                    "confidence": confidence
                })
            detections = fake_detections
        else:
            results = model(frame, conf=0.3)
            detections = []
            for det in results[0].boxes.data.tolist():
                score = det[4]
                class_id = int(det[5])
                class_name = CUSTOM_CLASS_NAMES[class_id] if class_id < len(CUSTOM_CLASS_NAMES) else f"class_{class_id}"
                class_name = class_name.replace("10", "T")
                detections.append({
                    "class_name": class_name,
                    "confidence": float(score)
                })
        with data_lock:
            latest_frame = frame
            latest_detections = detections
        time.sleep(0.1)


@app.route('/')
def index():
    with open('logs.html', 'r') as f:
        return f.read(), 200, {'Content-Type': 'text/html'}

@app.route('/info')
def info():
    with open('api.html', 'r') as f:
        return f.read(), 200, {'Content-Type': 'text/html'}

@app.route('/frame')
def get_frame():
    with data_lock:
        if latest_frame is None:
            message_frame = create_message_frame()
            ret, jpeg = cv2.imencode('.jpg', message_frame)
            if not ret:
                return Response(status=500)
            return Response(jpeg.tobytes(), mimetype='image/jpeg')
        frame = latest_frame.copy()
        if SIMULATION_MODE:
            annotated_frame = frame
        else:
            results = model(frame, conf=0.3)
            annotated_frame = results[0].plot()
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            return Response(status=500)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')



@app.route('/detections')
def get_detections():
    with data_lock:
        if not latest_detections:
            return jsonify([{"message": "Capture not started. Use POST /start_capture to begin."}])
        return jsonify(latest_detections)
    
@app.route('/unprocessed_frame')
def get_unprocessed_frame():
    with data_lock:
        if latest_frame is None:
            message_frame = create_message_frame()
            ret, jpeg = cv2.imencode('.jpg', message_frame)
            if not ret:
                return Response(status=500)
            return Response(jpeg.tobytes(), mimetype='image/jpeg')
        frame = latest_frame.copy()
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return Response(status=500)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')

    

@app.route('/model', methods=['GET', 'POST'])
def change_model():
    global model, model_name
    if request.method == 'POST':
        requested_model = request.json.get("model_name")
        if requested_model not in YOLO_MODEL_PATHS:
            return jsonify({"error": "Model not found"}), 404
        try:
            new_model = YOLO(YOLO_MODEL_PATHS[requested_model])
            with data_lock:
                model = new_model
                model_name = requested_model
            return jsonify({"status": f"Model changed to {requested_model}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"current_model": model_name, "available_models": list(YOLO_MODEL_PATHS.keys())})

@app.route('/start_capture', methods=['POST'])
def start_capture_endpoint():
    start_capture()
    logger.log(LogLevel.INFO, "Capture started via API", "start_capture_endpoint")
    return jsonify({"status": "Capture started"})

@app.route('/stop_capture', methods=['POST'])
def stop_capture_endpoint():
    stop_capture()
    logger.log(LogLevel.INFO, "Capture stopped via API", "stop_capture_endpoint")
    return jsonify({"status": "Capture stopped"})

@app.route('/health')
def health():
    model_loaded = False
    if not SIMULATION_MODE:
        try:
            model_loaded = model is not None
        except NameError:
            model_loaded = False
    
    return jsonify({
        "status": "healthy",
        "simulation_mode": SIMULATION_MODE,
        "model_loaded": model_loaded,
        "model_name": model_name if not SIMULATION_MODE else None
    })

@app.route('/detect', methods=['POST'])
def detect():
    """
    Endpoint to detect objects in an uploaded image.
    Returns detections as JSON.
    
    Expected request: multipart/form-data with 'image' file
    Returns: JSON with detections list
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image part in request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        if SIMULATION_MODE:
            import random
            num_detections = random.randint(1, 5)
            detections = []
            for _ in range(num_detections):
                class_name = random.choice(CUSTOM_CLASS_NAMES)
                confidence = round(random.uniform(0.5, 1.0), 2)
                detections.append({
                    "class_name": class_name,
                    "confidence": confidence
                })
        else:
            results = model(frame, conf=0.3)
            detections = []
            for det in results[0].boxes.data.tolist():
                score = det[4]
                class_id = int(det[5])
                class_name = CUSTOM_CLASS_NAMES[class_id] if class_id < len(CUSTOM_CLASS_NAMES) else f"class_{class_id}"
                class_name = class_name.replace("10", "T")
                detections.append({
                    "class_name": class_name,
                    "confidence": float(score)
                })
        
        return jsonify(detections)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====== LOG ENDPOINTS ======

@app.route('/logs', methods=['POST'])
def post_log():
    """Post a new log message"""
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

@app.route('/logs', methods=['GET'])
def get_logs():
    """Get all logs in JSON format organized by log level"""
    all_logs = logger.get_all_logs()
    return jsonify({
        "logs": all_logs
    })


@app.route('/logs/<level>', methods=['GET'])
def get_logs_by_level(level):
    """Get logs for a specific level"""
    try:
        from logger import LogLevel as LL
        log_level = LL[level.upper()]
        logs = logger.get_logs_by_level(log_level)
        return jsonify({
            "level": level.upper(),
            "logs": logs,
            "count": len(logs)
        })
    except KeyError:
        return jsonify({"error": f"Unknown log level: {level}"}), 400


@app.route('/logs/clear', methods=['POST'])
def clear_logs():
    """Clear all in-memory logs"""
    logger.log_buffer = {
        LogLevel.ERROR: [],
        LogLevel.WARNING: [],
        LogLevel.INFO: [],
        LogLevel.DETECTION: [],
        LogLevel.DECISION: []
    }
    return jsonify({"status": "Logs cleared"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7926, debug=True, use_reloader=False)


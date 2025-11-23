from flask import Flask, jsonify, request, Response
import threading
import time
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from ultralytics import YOLO
from config import YOLO_MODEL_PATHS, CUSTOM_CLASS_NAMES, colors, create_fake_image
from camera import Camera

app = Flask(__name__)

FAKE_MODE = False

camera = Camera(fake=FAKE_MODE)

if not FAKE_MODE:
    model_name = "yolo11n"
    try:
        model = YOLO(YOLO_MODEL_PATHS[model_name])
        print(f"Model loaded successfully from {YOLO_MODEL_PATHS[model_name]}")
    except Exception as e:
        print(f"Error loading model: {e}")

latest_frame = None
latest_detections = []
data_lock = threading.Lock()

def capture_thread():
    global latest_frame, latest_detections
    while True:
        frame = camera.capture()
        if FAKE_MODE:
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

# Start the background thread after initialization
threading.Thread(target=capture_thread, daemon=True).start()

@app.route('/frame')
def get_frame():
    with data_lock:
        if latest_frame is None:
            return Response(status=503)
        frame = latest_frame.copy()
        if FAKE_MODE:
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
        return jsonify(latest_detections)
    
@app.route('/unprocessed_frame')
def get_unprocessed_frame():
    with data_lock:
        if latest_frame is None:
            return Response(status=503)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7926, debug=True, use_reloader=False)


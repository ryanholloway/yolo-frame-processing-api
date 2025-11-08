from enum import Enum
from flask import Flask, jsonify, request, Response
import threading
import time
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from coco_class import CocoClass, coco_classes as COCO_NAMES


app = Flask(__name__)

YOLO_MODEL_PATHS ={
    "yolo11n":"best.pt"
}

CUSTOM_CLASS_NAMES = [
    '10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', 
    '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', 
    '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S', 
    'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 
    'QC', 'QD', 'QH', 'QS'
]


FAKE_MODE = False

if not FAKE_MODE:
    from picamera2 import Picamera2
    from ultralytics import YOLO


def initialize_camera():
    camera = Picamera2()
    config = camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 640)})
    camera.configure(config)
    camera.start()
    time.sleep(2)
    return camera

if not FAKE_MODE:
    picam2 = initialize_camera()
    model_name = "yolo11n"
    try:
        model = YOLO(YOLO_MODEL_PATHS[model_name])
        print(f"Model loaded successfully from {YOLO_MODEL_PATHS[model_name]}")
    except Exception as e:
        print(f"Error loading model: {e}")

latest_frame = None
latest_detections = []
data_lock = threading.Lock()

color_index = 0
colors = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0)   # Cyan
]

def create_fake_image():
    global color_index, colors
    height, width = 360, 640
    gradiantLevel = np.random.rand()
    vertical_gradient = np.tile(np.linspace(0, 255, height, dtype=np.uint8), (width, 1)).T
    horizontal_gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    
    combined_gradient = cv2.addWeighted(vertical_gradient, gradiantLevel, horizontal_gradient, gradiantLevel, 0)
    fake_image = cv2.merge([combined_gradient, vertical_gradient, horizontal_gradient])
    
    color = colors[1]
    
    text_x = 50
    text_y = 240

    cv2.putText(fake_image, 'Fake Frame', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)
    return fake_image


@app.route('/frame')
def get_frame():
    if FAKE_MODE:
        fake_image = create_fake_image()
        ret, jpeg = cv2.imencode('.jpg', fake_image)
        if not ret:
            return Response(status=500)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
    else:
        frame = picam2.capture_array()
        results = model(frame, conf=0.3)
        annotated_frame = results[0].plot()
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            return Response(status=500)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/detections')
def get_detections():
    if FAKE_MODE:
        dummy_detections = [
            {"class_name": "person", "confidence": 0.99},
            {"class_name": "car", "confidence": 0.85}
        ]
        return jsonify(dummy_detections)
    else:
        frame = picam2.capture_array()
        results = model(frame, conf=0.3)
        detections = []
        for det in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            class_id = int(class_id)
            class_name = CUSTOM_CLASS_NAMES[class_id] if class_id < len(CUSTOM_CLASS_NAMES) else f"class_{class_id}"
            class_name = class_name.replace("10", "T")
            detections.append({
                "class_name": class_name,
                "confidence": float(score)
            })
        return jsonify(detections)

@app.route('/unprocessed_frame')
def get_unprocessed_frame():
    frame = picam2.capture_array()
    ret, jpeg = cv2.imencode('.jpg', frame)
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


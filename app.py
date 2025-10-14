from enum import Enum
from flask import Flask, jsonify, request, Response
import threading
import time
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

app = Flask(__name__)

YOLO_MODEL_PATHS ={
    "yollo11n":"yolo11n.pt",
    "yollo11s":"yolo11s.pt",
}

FAKE_MODE = True

if not FAKE_MODE:
    from picamera2 import Picamera2
    from ultralytics import YOLO

    

class CocoClasses(Enum):
    person = 0
    bicycle = 1
    car = 2
    motorcycle = 3
    airplane = 4
    bus = 5
    train = 6
    truck = 7
    boat = 8
    traffic_light = 9
    fire_hydrant = 10
    stop_sign = 11
    parking_meter = 12
    bench = 13
    bird = 14
    cat = 15
    dog = 16
    horse = 17
    sheep = 18
    cow = 19
    elephant = 20
    bear = 21
    zebra = 22
    giraffe = 23
    backpack = 24
    umbrella = 25
    handbag = 26
    tie = 27
    suitcase = 28
    frisbee = 29
    skis = 30
    snowboard = 31
    sports_ball = 32
    kite = 33
    baseball_bat = 34
    baseball_glove = 35
    skateboard = 36
    surfboard = 37
    tennis_racket = 38
    bottle = 39
    wine_glass = 40
    cup = 41
    fork = 42
    knife = 43
    spoon = 44
    bowl = 45
    banana = 46
    apple = 47
    sandwich = 48
    orange = 49
    broccoli = 50
    carrot = 51
    hot_dog = 52
    pizza = 53
    donut = 54
    cake = 55
    chair = 56
    couch = 57
    potted_plant = 58
    bed = 59
    dining_table = 60
    toilet = 61
    tv = 62
    laptop = 63
    mouse = 64
    remote = 65
    keyboard = 66
    cell_phone = 67
    microwave = 68
    oven = 69
    toaster = 70
    sink = 71
    refrigerator = 72
    book = 73
    clock = 74
    vase = 75
    scissors = 76
    teddy_bear=77 
    hair_drier=78
    toothbrush=79
    void=80

COCO_NAMES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush", "void"
]

if not FAKE_MODE:
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    model_name = "yollo11n"
    model = YOLO(YOLO_MODEL_PATHS[model_name])

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
    height, width = 480, 640
    
    vertical_gradient = np.tile(np.linspace(0, 255, height, dtype=np.uint8), (width, 1)).T
    horizontal_gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    
    combined_gradient = cv2.addWeighted(vertical_gradient, 0.5, horizontal_gradient, 0.5, 0)
    fake_image = cv2.merge([combined_gradient, vertical_gradient, horizontal_gradient])
    
    color = colors[color_index % len(colors)]
    
    text_x = 50 + (color_index * 30) % (width - 200)
    text_y = 240 + (color_index * 20) % (height - 100)
    
    cv2.putText(fake_image, 'Fake Frame', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)
    
    color_index += 1
    return fake_image

def fake_detection_loop(data_lock, stop_flag):
    global latest_frame, latest_detections

    dummy_detections = [
        {"class_name": "person", "confidence": 0.99},
        {"class_name": "car", "confidence": 0.85}
    ]

    update_interval = 0.1

    while not stop_flag.is_set():
        start_time = time.time()
        
        fake_image = create_fake_image()
        ret, jpeg = cv2.imencode('.jpg', fake_image)
        frame_bytes = jpeg.tobytes()

        with data_lock:
            latest_frame = frame_bytes
            latest_detections = dummy_detections
        
        elapsed = time.time() - start_time
        time.sleep(max(0, update_interval - elapsed))

def detection_loop():
    global latest_frame, latest_detections, model
    while True:
        frame = picam2.capture_array()
        results = model(frame)
        annotated_frame = results[0].plot()
        detections = []
        for det in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            class_id = int(class_id)
            detections.append({
                "class_name": COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else f"class_{class_id}",
                "confidence": float(score)
            })
        frame_bytes = cv2.imencode('.jpg', annotated_frame)[1].tobytes()
        with data_lock:
            latest_frame = frame_bytes
            latest_detections = detections
        time.sleep(0.05)


@app.route('/frame')
def get_frame():
    with data_lock:
        if latest_frame is None:
            return Response(status=503)
        return Response(latest_frame, mimetype='image/jpeg')

@app.route('/detections')
def get_detections():
    with data_lock:
        return jsonify(latest_detections)
    

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
    stop_flag = threading.Event()

    if FAKE_MODE:
        threading.Thread(target=fake_detection_loop, args=(data_lock, stop_flag), daemon=True).start()
    else:
        threading.Thread(target=detection_loop, args=(data_lock, stop_flag), daemon=True).start()

    app.run(host='0.0.0.0', port=7926, debug=True, use_reloader=True)


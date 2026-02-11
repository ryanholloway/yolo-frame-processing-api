import random
from ultralytics import YOLO
from app.config import YOLO_MODEL_PATHS, CUSTOM_CLASS_NAMES


class DetectionService:
    def __init__(self, simulation_mode=False, model_name="yolo11n"):
        self.simulation_mode = simulation_mode
        self.model = None
        self.model_name = model_name
        
        if not simulation_mode:
            try:
                self.model = YOLO(YOLO_MODEL_PATHS[model_name])
                print(f"Model loaded successfully from {YOLO_MODEL_PATHS[model_name]}")
            except Exception as e:
                print(f"Error loading model: {e}")

    def detect(self, frame, conf=0.3):
        if self.simulation_mode:
            return self._simulate_detection()
        else:
            return self._yolo_detection(frame, conf)

    def _simulate_detection(self):
        num_detections = random.randint(1, 5)
        detections = []
        for _ in range(num_detections):
            class_name = random.choice(CUSTOM_CLASS_NAMES)
            confidence = round(random.uniform(0.5, 1.0), 2)
            detections.append({
                "class_name": class_name,
                "confidence": confidence
            })
        return detections

    def _yolo_detection(self, frame, conf):
        results = self.model(frame, conf=conf)
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
        return detections

    def annotate_frame(self, frame, conf=0.3):
        if self.simulation_mode:
            return frame
        else:
            results = self.model(frame, conf=conf)
            return results[0].plot()

    def change_model(self, model_name):
        if model_name not in YOLO_MODEL_PATHS:
            raise ValueError("Model not found")
        try:
            new_model = YOLO(YOLO_MODEL_PATHS[model_name])
            self.model = new_model
            self.model_name = model_name
            return True
        except Exception as e:
            raise Exception(f"Error changing model: {str(e)}")

    def get_current_model(self):
        return self.model_name

    def get_available_models(self):
        return list(YOLO_MODEL_PATHS.keys())

    def is_model_loaded(self):
        return self.model is not None

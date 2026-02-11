import threading
import time
from app.utils.camera import Camera


class CaptureService:
    def __init__(self, simulation_mode, detection_service, logger):
        self.simulation_mode = simulation_mode
        self.detection_service = detection_service
        self.logger = logger
        self.camera = Camera(simulation_mode=simulation_mode)
        
        self.latest_frame = None
        self.latest_detections = []
        self.data_lock = threading.Lock()
        self.is_capturing = False
        self.capture_thread_instance = None

    def start_capture(self):
        if not self.is_capturing:
            self.is_capturing = True
            self.capture_thread_instance = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread_instance.start()

    def stop_capture(self):
        self.is_capturing = False

    def _capture_loop(self):
        while self.is_capturing:
            frame = self.camera.capture()
            detections = self.detection_service.detect(frame, conf=0.3)
            
            with self.data_lock:
                self.latest_frame = frame
                self.latest_detections = detections
            
            time.sleep(0.1)

    def get_latest_frame(self):
        with self.data_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_latest_detections(self):
        with self.data_lock:
            return self.latest_detections[:]

    def has_frame(self):
        with self.data_lock:
            return self.latest_frame is not None

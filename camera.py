from config import CAMERA_CONFIG, create_fake_image

class Camera:
    def __init__(self, simulation_mode=False):
        self.simulation_mode = simulation_mode
        if not simulation_mode:
            try:
                from picamera2 import Picamera2
                import time
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(main=CAMERA_CONFIG)
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)
            except (ImportError, RuntimeError) as e:
                print(f"Camera initialization failed: {e}. Falling back to simulation mode")
                self.simulation_mode = True
        if self.simulation_mode:
            pass

    def capture(self):
        if self.simulation_mode:
            return create_fake_image()
        else:
            return self.camera.capture_array()
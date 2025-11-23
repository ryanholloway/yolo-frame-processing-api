from config import CAMERA_CONFIG, create_fake_image

class Camera:
    def __init__(self, fake=False):
        self.fake = fake
        if not fake:
            try:
                from picamera2 import Picamera2
                import time
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(main=CAMERA_CONFIG)
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)
            except ImportError:
                print("picamera2 not available, falling back to fake mode")
                self.fake = True
        if self.fake:
            pass

    def capture(self):
        if self.fake:
            return create_fake_image()
        else:
            return self.camera.capture_array()
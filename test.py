from flask import Flask, Response
from picamera2 import Picamera2
import cv2

app = Flask(__name__)

import time

def initialize_camera():
    camera = Picamera2()
    config = camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 360)})
    camera.configure(config)
    camera.start()
    time.sleep(2)
    return camera

picam2 = initialize_camera()


@app.route('/frame')
def get_frame():
    frame = picam2.capture_array()
    ret, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7926, debug=True, use_reloader=False)

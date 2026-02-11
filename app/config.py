import cv2
import numpy as np


class Config:
    SIMULATION_MODE = False
    DEFAULT_MODEL = "yolo11n"
    HOST = '0.0.0.0'
    PORT = 7926
    DEBUG = True


YOLO_MODEL_PATHS = {
    "yolo11n": "models/best.pt"
}

CUSTOM_CLASS_NAMES = [
    '10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S',
    '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S',
    '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S',
    'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS',
    'QC', 'QD', 'QH', 'QS'
]

COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0)
]

IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 640

CAMERA_CONFIG = {"format": "RGB888", "size": (IMAGE_WIDTH, IMAGE_HEIGHT)}


def create_fake_image():
    height, width = IMAGE_HEIGHT, IMAGE_WIDTH
    gradiant_level = np.random.rand()
    vertical_gradient = np.tile(np.linspace(0, 255, height, dtype=np.uint8), (width, 1)).T
    horizontal_gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    
    combined_gradient = cv2.addWeighted(vertical_gradient, gradiant_level, horizontal_gradient, gradiant_level, 0)
    fake_image = cv2.merge([combined_gradient, vertical_gradient, horizontal_gradient])
    
    color = COLORS[1]
    text_x = 50
    text_y = 240
    
    cv2.putText(fake_image, 'Fake Frame', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)
    return fake_image


def create_message_frame(message="Capture not started. Use POST /start_capture to begin."):
    height, width = IMAGE_HEIGHT, IMAGE_WIDTH
    message_frame = np.zeros((height, width, 3), dtype=np.uint8)
    message_frame[:] = (50, 50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(message_frame, message, (text_x, text_y), font, font_scale, color, thickness)
    return message_frame

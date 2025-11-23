YOLO_MODEL_PATHS = {
    "yolo11n": "best.pt"
}

CUSTOM_CLASS_NAMES = [
    '10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', 
    '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', 
    '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S', 
    'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 
    'QC', 'QD', 'QH', 'QS'
]

colors = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0)   # Cyan
]

IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 640

CAMERA_CONFIG = {"format": "RGB888", "size": (IMAGE_WIDTH, IMAGE_HEIGHT)}

def create_fake_image():
    import cv2
    import numpy as np
    height, width = IMAGE_HEIGHT, IMAGE_WIDTH
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
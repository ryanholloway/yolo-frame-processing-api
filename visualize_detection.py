#!/usr/bin/env python3
"""
Script to detect objects in an image and display the annotated result.
Usage: python visualize_detection.py <image_path>
"""

import sys
import requests
import base64
import cv2
import numpy as np
from pathlib import Path


def detect_and_visualize(image_path, server_url="http://localhost:7926"):
    """Send image to API and display annotated result."""
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"Error: File '{image_path}' not found")
        return
    
    # Read and send image
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(
            f"{server_url}/detect?annotate=true",
            files=files
        )
    
    if response.status_code != 200:
        print(f"Error: Server returned {response.status_code}")
        print(response.text)
        return
    
    data = response.json()
    
    # Print detections
    print("\n🎯 Detected Objects:")
    print("-" * 50)
    for detection in data.get('detections', []):
        class_name = detection['class_name']
        confidence = detection['confidence'] * 100
        print(f"  {class_name}: {confidence:.1f}%")
    print("-" * 50)
    
    # Display annotated image if available
    if 'annotated_image' in data:
        print("\n📸 Displaying annotated image... (Press any key to close)")
        
        # Decode base64 image
        image_bytes = base64.b64decode(data['annotated_image'])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Display
        cv2.imshow('YOLO Detection Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Optionally save
        save_path = Path(image_path).stem + '_annotated.jpg'
        cv2.imwrite(save_path, img)
        print(f"✅ Saved annotated image to: {save_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_detection.py <image_path> [server_url]")
        print("Example: python visualize_detection.py card.jpg")
        print("Example: python visualize_detection.py card.jpg http://192.168.1.100:7926")
        sys.exit(1)
    
    image_path = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:7926"
    
    detect_and_visualize(image_path, server_url)

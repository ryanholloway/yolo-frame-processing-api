# YOLO Object Detection Flask API

This repository contains a Flask-based web API for real-time object detection using YOLO models. It supports both a fake mode for testing without camera hardware and a live mode using a Raspberry Pi camera and the Ultralytics YOLO implementation.

## Features

- Real-time frame capture and object detection with YOLO models.
- Endpoint to get the latest annotated frame as JPEG.
- Endpoint to get the latest detection results as JSON.
- Ability to switch between supported YOLO models dynamically.
- Fake mode to simulate camera capture and detections without hardware for development/testing.

## Contents

- `app.py`: Main Flask application implementing the API, detection loops, and fake image generation.
- `requirements.txt`: Python dependencies for the project.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ryanholloway/yolo-frame-processing-api
cd yolo-frame-processing-api
```
2. Create a Python virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:

```
pip install -r requirements.txt
```
## Configuration

- By default, `FAKE_MODE` is set to `True` in `app.py` for development without a real camera.
- To use a Raspberry Pi camera, set `FAKE_MODE` to `False` and ensure the `picamera2` and `ultralytics` packages are installed and functional.
- The YOLO model paths are defined in the `YOLO_MODEL_PATHS` dictionary. Add or modify models as needed.

## Usage

Run the Flask app:

``` 
python app.py
```

The server will start on `http://0.0.0.0:7926`.

### API Endpoints

- `GET /frame`  
  Returns the latest annotated frame as a JPEG image. Returns 503 if no frame is available.

- `GET /detections`  
  Returns JSON array of the latest detected objects with class names and confidence scores.

- `GET /model`  
  Returns the current YOLO model name and available models.

- `POST /model`  
  Switches the YOLO model used for detection.  
  Request JSON example:  
```json
{
"model_name": "yollo11n"
}
```
Returns success status or error if model not found.

## Overview

The app uses threading to continuously capture frames and run object detection asynchronously:

- In fake mode, a synthetic gradient image with "Fake Frame" text is generated with dummy detections.
- In live mode, frames are captured from the Raspberry Pi camera and processed with YOLO.

The latest frame and detections are kept thread-safe with a lock and served via the Flask endpoints.

## Dependencies

See `requirements.txt` for required Python packages. Key dependencies include:

- Flask
- OpenCV (`opencv-python`)
- NumPy
- Ultralytics YOLO (only required in live mode)
- Picamera2 (only required in live mode)

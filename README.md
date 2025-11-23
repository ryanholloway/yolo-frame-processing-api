# YOLO Object Detection Flask API

This repository contains a Flask-based web API for real-time object detection using YOLO models. It supports both a simulation mode for testing without camera hardware and a live mode using a Raspberry Pi camera and the Ultralytics YOLO implementation.

## Features

- Real-time frame capture and object detection with YOLO models.
- Endpoint to get the latest annotated frame as JPEG.
- Endpoint to get the latest detection results as JSON.
- Endpoint to get the latest unprocessed frame as JPEG.
- Ability to switch between supported YOLO models dynamically.
- Simulation mode to simulate camera capture and detections without hardware for development/testing.
- Background caching of frames and detections for efficient API responses.

## Contents

- `app.py`: Main Flask application implementing the API, detection loops, and fake image generation.
- `config.py`: Configuration file containing model paths, class names, colors, and utility functions.
- `camera.py`: Camera abstraction for device-agnostic frame capture (supports Raspberry Pi and fallback to fake mode).
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

- By default, `SIMULATION_MODE` is set to `False` in `app.py` for live camera mode.
- To use simulation mode for development without a real camera, set `SIMULATION_MODE` to `True`.
- If `picamera2` is not available (e.g., on non-Raspberry Pi devices), the camera will automatically fall back to simulation mode.
- The YOLO model paths are defined in the `YOLO_MODEL_PATHS` dictionary in `config.py`. Add or modify models as needed.
- Custom class names for detections are defined in `CUSTOM_CLASS_NAMES` in `config.py`.

## Usage

Run the Flask app:

```
sudo -E venv/bin/python app.py
```

The server will start on `http://0.0.0.0:7926`.

### API Endpoints

- `GET /`  
  Returns an HTML page with API documentation.

- `GET /frame`  
  Returns the latest annotated frame as a JPEG image with detection bounding boxes. If capture hasn't started, returns a message image.

- `GET /detections`  
  Returns JSON array of the latest detected objects with class names and confidence scores. If capture hasn't started, returns a message.

- `GET /unprocessed_frame`  
  Returns the latest unprocessed frame as a JPEG image without annotations. If capture hasn't started, returns a message image.

- `GET /model`  
  Returns the current YOLO model name and available models.

- `POST /model`  
  Switches the YOLO model used for detection.  
  Request JSON example:

```json
{
  "model_name": "yolo11n"
}
```

Returns success status or error if model not found.

- `POST /start_capture`  
  Starts the frame capture and detection process.

- `POST /stop_capture`  
  Stops the frame capture and detection process.

## Overview

The app can start frame capture and object detection on demand via the `/start_capture` endpoint. By default, no capture or detection occurs to conserve resources:

- In simulation mode, a synthetic gradient image with "Fake Frame" text is generated with dummy detections.
- In live mode, frames are captured from the Raspberry Pi camera and processed with YOLO.

The latest frame and detections are kept thread-safe with a lock and served via the Flask endpoints. This allows `/detections` and `/frame` to be called separately without redundant processing.

## Dependencies

See `requirements.txt` for required Python packages. Key dependencies include:

- Flask
- OpenCV (`opencv-python`)
- NumPy
- Ultralytics YOLO (only required in live mode)
- Picamera2 (only required in live mode)

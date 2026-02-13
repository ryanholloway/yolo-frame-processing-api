import cv2
import numpy as np
from flask import Blueprint, jsonify, request, Response
from app.config import create_message_frame

bp = Blueprint('detection', __name__)


@bp.route('/frame')
def get_frame():
    from app import get_capture_service, get_detection_service
    
    capture_service = get_capture_service()
    detection_service = get_detection_service()
    
    if not capture_service.has_frame():
        message_frame = create_message_frame()
        ret, jpeg = cv2.imencode('.jpg', message_frame)
        if not ret:
            return Response(status=500)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
    
    frame = capture_service.get_latest_frame()
    annotated_frame = detection_service.annotate_frame(frame, conf=0.3)
    
    ret, jpeg = cv2.imencode('.jpg', annotated_frame)
    if not ret:
        return Response(status=500)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@bp.route('/unprocessed_frame')
def get_unprocessed_frame():
    from app import get_capture_service
    
    capture_service = get_capture_service()
    
    if not capture_service.has_frame():
        message_frame = create_message_frame()
        ret, jpeg = cv2.imencode('.jpg', message_frame)
        if not ret:
            return Response(status=500)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
    
    frame = capture_service.get_latest_frame()
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        return Response(status=500)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@bp.route('/detections')
def get_detections():
    from app import get_capture_service
    
    capture_service = get_capture_service()
    
    detections = capture_service.get_latest_detections()
    if not detections:
        return jsonify([{"message": "Capture not started. Use POST /start_capture to begin."}])
    return jsonify(detections)


@bp.route('/detect', methods=['POST'])
def detect():
    from app import get_detection_service
    import base64
    
    if 'image' not in request.files:
        return jsonify({"error": "No image part in request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        detection_service = get_detection_service()
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        detections = detection_service.detect(frame, conf=0.3)
        
        # Check if annotated image is requested
        include_image = request.args.get('annotate', 'false').lower() == 'true'
        
        if include_image:
            annotated_frame = detection_service.annotate_frame(frame, conf=0.3)
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if ret:
                # Encode image to base64 for JSON response
                image_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                return jsonify({
                    "detections": detections,
                    "annotated_image": image_base64,
                    "image_format": "jpeg"
                })
        
        return jsonify(detections)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/model', methods=['GET', 'POST'])
def change_model():
    from app import get_detection_service
    
    detection_service = get_detection_service()
    
    if request.method == 'POST':
        requested_model = request.json.get("model_name")
        try:
            detection_service.change_model(requested_model)
            return jsonify({"status": f"Model changed to {requested_model}"})
        except ValueError:
            return jsonify({"error": "Model not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({
        "current_model": detection_service.get_current_model(),
        "available_models": detection_service.get_available_models()
    })


@bp.route('/detection-service', methods=['GET', 'POST'])
def switch_detection_service():
    """Switch between Hailo and YOLO detection services at runtime"""
    from app import set_detection_service, get_detection_service
    
    if request.method == 'GET':
        # Get current service info
        detection_service = get_detection_service()
        return jsonify({
            "current_service": detection_service.model_name if hasattr(detection_service, 'model_name') else "unknown",
            "available_services": ["hailo", "yolo"],
            "yolo_models": ["yolo11n"],
            "hailo_model": "yolov8m.hef"
        })
    
    # POST - switch service
    try:
        data = request.json
        service_type = data.get('service')
        
        if not service_type:
            return jsonify({"error": "Missing 'service' parameter. Use 'hailo' or 'yolo'"}), 400
        
        result = set_detection_service(
            service_type,
            model=data.get('model', 'yolo11n'),
            simulation_mode=data.get('simulation_mode', False)
        )
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to switch service: {str(e)}"}), 500


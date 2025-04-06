from flask import Flask, Response, send_from_directory, jsonify
import cv2
from stream_server import get_camera

app = Flask(__name__, static_folder='.')
camera = None

def get_camera_instance():
    global camera
    if camera is None:
        camera = get_camera()
    return camera

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

def gen(camera):
    while True:
        frame, detection_info = camera.get_frame()
        if frame is not None:
            # Store the latest detection info
            app.detection_info = detection_info
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    camera = get_camera_instance()
    response = Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/detection_info')
def detection_info():
    response = jsonify(getattr(app, 'detection_info', {
        "num_persons": 0,
        "num_males": 0,
        "num_females": 0,
        "status": "No person detected",
        "coverage_ratio": 0,
        "coverage_status": "Coverage: 0.00"
    }))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

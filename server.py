from flask import Flask, request, Response
import cv2
import numpy as np
from woman_info1 import getFaceBox, net  # Ensure net model is loaded in woman_info.py

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    img_data = request.data
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run detection
    frame, bboxes = getFaceBox(net, frame)
    _, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

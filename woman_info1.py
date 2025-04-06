import cv2
import numpy as np
import time
from flask import Flask, request, Response, jsonify

app = Flask(__name__)

# Global variables to store the latest frame and gender counts
latest_frame = None
latest_info = {
    "num_males": 0,
    "num_females": 0,
    "status": "No person detected"
}

# Define the face detection function
def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, bboxes

# Load face and gender detection models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

@app.route('/upload', methods=['POST'])
def upload_image():
    global latest_frame, latest_info

    # Receive and process the image
    img_data = request.data
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Detect faces
    frameFace, bboxes = getFaceBox(faceNet, frame)
    numMales = 0
    numFemales = 0
    numPersons = len(bboxes)
    
    # Draw person count on the frame
    cv2.putText(frameFace, f'Person Count: {numPersons}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Process gender for each detected face
    for bbox in bboxes:
        face = frame[max(0, bbox[1] - 20):min(bbox[3] + 20, frame.shape[0] - 1),
                     max(0, bbox[0] - 20):min(bbox[2] + 20, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        if gender == 'Male':
            numMales += 1
        else:
            numFemales += 1

        cv2.putText(frameFace, f'{gender}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Display counts of males and females
    cv2.putText(frameFace, f'Males: {numMales}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frameFace, f'Females: {numFemales}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # Determine status based on counts
    status = "No person detected"
    if numFemales == 1:
        if numMales >= 1:
            status = "Woman is surrounded by men"
            cv2.putText(frameFace, status, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            status = "Woman is alone"
            cv2.putText(frameFace, status, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Update global variables for latest frame and info
    latest_frame = frameFace
    latest_info = {
        "num_males": numMales,
        "num_females": numFemales,
        "status": status
    }

    # Return a simple response
    return "Image processed", 200

# Route to display the latest processed image
@app.route('/preview')
def preview_image():
    global latest_frame
    if latest_frame is None:
        return "No image available", 404
    _, jpeg = cv2.imencode('.jpg', latest_frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

# Route to get the latest info in JSON format
@app.route('/info')
def latest_info_json():
    return jsonify(latest_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

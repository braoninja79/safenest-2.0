import cv2
import time
import argparse
import json
import numpy as np

def getFaceBox(net, frame, conf_threshold=0.75):
    """
    Detect faces in the frame using a pre-trained DNN model.
    Returns an annotated copy of the frame and a list of bounding boxes.
    """
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

def check_coverage(frame, grid_rows=3, grid_cols=3, variance_threshold=100.0):
    """
    Divides the frame into a grid and calculates the fraction of grid cells that are "covered"
    (i.e. have low variance of the Laplacian). Returns the overall coverage ratio.
    """
    height, width = frame.shape[:2]
    cell_height = height // grid_rows
    cell_width = width // grid_cols
    covered_cells = 0
    total_cells = grid_rows * grid_cols

    for i in range(grid_rows):
        for j in range(grid_cols):
            y1 = i * cell_height
            y2 = (i + 1) * cell_height if i < grid_rows - 1 else height
            x1 = j * cell_width
            x2 = (j + 1) * cell_width if j < grid_cols - 1 else width
            cell = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            var = laplacian.var()
            if var < variance_threshold:
                covered_cells += 1

    coverage_ratio = covered_cells / total_cells
    return coverage_ratio

# Argument parsing: optionally process an image file; defaults to webcam.
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to image file or leave empty for webcam')
args = parser.parse_args()

# Model paths for face detection and gender classification.
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

# Load models.
faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Start video capture from provided image or webcam.
video = cv2.VideoCapture(args.image if args.image else 0)
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

padding = 20
scale_factor = 1.5  # Scale factor for display

while True:
    t = time.time()
    ret, frame = video.read()
    if not ret:
        print("No frame captured from video.")
        break

    # Detect faces in the frame.
    frameFace, bboxes = getFaceBox(faceNet, frame)
    
    # Initialize gender counters.
    numMales = 0
    numFemales = 0
    numPersons = len(bboxes)
    cv2.putText(frameFace, f'Person Count: {numPersons}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    if not bboxes:
        print("No face detected, checking next frame.")
    
    # Process each detected face for gender classification.
    for bbox in bboxes:
        face_region = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                            max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face_region, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]  # Determine gender

        if gender == 'Male':
            numMales += 1
        else:
            numFemales += 1

        cv2.putText(frameFace, f'{gender}', (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        print(f'Gender: {gender}')

    cv2.putText(frameFace, f'Males: {numMales}', (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frameFace, f'Females: {numFemales}', (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # Determine status based on counts.
    status = "No person detected"
    if numPersons >= 1:
        if numFemales == 1:
            if numMales >= 3:
                status = "Woman is surrounded by men"
            else:
                status = "Woman is alone"
        elif numPersons >= 2:
            status = "Multiple persons detected"
        cv2.putText(frameFace, status, (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Calculate the coverage ratio for the frame.
    coverage_ratio = check_coverage(frame, grid_rows=3, grid_cols=3, variance_threshold=100.0)
    # Check if 100% of the display is covered (or nearly 100%).
    if coverage_ratio >= 0.99:
        cv2.putText(frameFace, "100% display is covered!", (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        print("Alert: 100% display is covered! Coverage ratio:", coverage_ratio)
    elif coverage_ratio >= 0.4:
        cv2.putText(frameFace, "Screen covered over 40%!", (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        print("Alert: Screen is covered over 40%! Coverage ratio:", coverage_ratio)
    else:
        cv2.putText(frameFace, f"Coverage: {coverage_ratio:.2f}", (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Print unified detection info.
    detection_info = {
        "num_persons": numPersons,
        "num_males": numMales,
        "num_females": numFemales,
        "status": status,
        "coverage_ratio": coverage_ratio
    }
    print(json.dumps(detection_info))

    # Resize frame for better display.
    height, width = frameFace.shape[:2]
    frameFace_resized = cv2.resize(frameFace, (int(width * scale_factor), int(height * scale_factor)))

    cv2.imshow("Output", frameFace_resized)
    print("Time: {:.3f}".format(time.time() - t))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

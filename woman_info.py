import cv2
import time
import argparse

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

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to image file or leave empty for webcam')
args = parser.parse_args()

# Model paths
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Start video capture
video = cv2.VideoCapture(args.image if args.image else 0)
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

padding = 20
scale_factor = 1.5  # Factor by which to scale up the display frame

while True:
    t = time.time()
    hasFrame, frame = video.read()
    if not hasFrame:
        print("No frame captured from video.")
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)

    # Initialize gender counts
    numMales = 0
    numFemales = 0

    numPersons = len(bboxes)
    cv2.putText(frameFace, f'Person Count: {numPersons}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if not bboxes:
        print("No face detected, checking next frame.")
    
    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]  # Determine gender

        # Increment male or female count
        if gender == 'Male':
            numMales += 1
        else:
            numFemales += 1

        cv2.putText(frameFace, f'{gender}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        print(f'Gender: {gender}')

    # Display the counts of males and females
    cv2.putText(frameFace, f'Males: {numMales}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frameFace, f'Females: {numFemales}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # Check conditions and display appropriate messages
    if numFemales == 1:
        if numMales >= 1:
            cv2.putText(frameFace, 'Woman is surrounded by men', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print('Woman is surrounded by men')
        else:
            cv2.putText(frameFace, 'Woman is alone', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print('Woman is alone')

    # Print counts to the console
    print(f'Males: {numMales}, Females: {numFemales}')

    # Scale up the frame for display
    height, width = frameFace.shape[:2]
    frameFace_resized = cv2.resize(frameFace, (int(width * scale_factor), int(height * scale_factor)))

    cv2.imshow("Gender Detection", frameFace_resized)
    print("Time: {:.3f}".format(time.time() - t))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


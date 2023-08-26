import cv2   # for image processing
import math
import argparse  # Import the argparse library for command-line argument handling


# Function to highlight faces in an image
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]  # height
    frameWidth = frameOpencvDnn.shape[1]  # width
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)
    # blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    net.setInput(blob)  # Set the input for the neural network

    detections = net.forward()  # Perform forward pass through the network

    faceBoxes = []

    # Loop through the detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (255, 255,255), int(
                round(frameHeight/150)), 8)  # rect around faces

    # Return the modified frame and detected face boxes
    return frameOpencvDnn, faceBoxes


# Parse command - line arguments
parser = argparse.ArgumentParser()
# Use the required argument for specifying the image path
parser.add_argument('--image', required=True)
args = parser.parse_args()


# Define paths to the required model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


# Define constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-5)', '(6-10)', '(11-15)', '(15-20)', '(21-25)',
           '(26-32)', '(33-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


# Load neural network models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


image_path = args.image  # Get the image path from command-line argument
frame = cv2.imread(image_path)  # Load the image from the specified path

padding = 20


# Highlight faces and perform age and gender prediction
resultImg, faceBoxes = highlightFace(faceNet, frame)
if not faceBoxes:
    print("No face detected")


# Loop through the detected faces
for faceBox in faceBoxes:
    # Extract the face region from the frame
    face = frame[max(0, faceBox[1] - padding):
                 min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

    blob = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict Gender
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    print(f'Gender: {gender}')

    # Predict age
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    print(f'Age: {age[1:-1]} years')

    # Display age and gender on the result image
    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Detecting age and gender", resultImg)
    cv2.waitKey(0)
cv2.destroyAllWindows()

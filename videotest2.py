import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

# Load model
model = load_model("/teamspace/studios/this_studio/Emotion-Detection-openCV/models/best_model.h5")

# Load the Haar cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/teamspace/studios/this_studio/Emotion-Detection-openCV/haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # Capture frame and return boolean value and captured image
    if not ret:
        continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # Cropping region of interest i.e. face area from image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # Find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    if cv2.waitKey(10) & 0xFF == ord('q'):  # Wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()

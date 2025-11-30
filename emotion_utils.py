import cv2
import numpy as np
from keras.models import load_model
import streamlit as st

# --- Load Emotion Detection Model ---
@st.cache_resource
def load_emotion_model():
    return load_model('CNN_Model_acc_75.h5')

emotion_model = load_emotion_model()
img_shape = 48
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Preprocess Frame for Emotion Detection ---
def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotions_detected = []

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        face_roi = cv2.resize(roi_color, (img_shape, img_shape))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / float(img_shape)
        predictions = emotion_model.predict(face_roi)
        emotion = emotion_labels[np.argmax(predictions[0])]
        emotions_detected.append(emotion)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame, emotions_detected
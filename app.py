from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import model_from_json
import os

app = Flask(__name__)

# Load model architecture from JSON file
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the pre-trained model from JSON
loaded_model = model_from_json(loaded_model_json)

# Load model weights from H5 file
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Define emotion labels for mapping predictions
emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to predict emotion from facial image
def predict_emotion(face_image):
    """
    Predicts emotion from the facial image.

    Parameters:
        face_image (numpy.ndarray): Grayscale facial image of shape (48, 48).

    Returns:
        str: Predicted emotion label.
    """
    # Resize the face image to match the model input size
    face_image = cv2.resize(face_image, (48, 48))
    face_image = face_image.astype("float") / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(face_image)
    max_index = np.argmax(prediction)
    emotion = emotion_labels[max_index]
    return emotion

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    """Generate frame for video streaming."""
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            # Crop the face region for emotion prediction
            face_roi = gray[y:y + h, x:x + w]

            # Predict emotion from the cropped face region
            emotion = predict_emotion(face_roi)

            # Display the predicted emotion on the frame
            cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
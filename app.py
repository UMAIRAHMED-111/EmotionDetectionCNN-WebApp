from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from keras.models import model_from_json
import os
import base64
import io
from datetime import datetime

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
saved_frames = []

emotion_count = {label: 0 for label in emotion_labels}

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def predict_emotion(face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = face_image.astype("float") / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    prediction = loaded_model.predict(face_image)
    max_index = np.argmax(prediction)
    emotion = emotion_labels[max_index]
    return emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    encoded_image = data['image'].split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    image_buffer = io.BytesIO(decoded_image)
    image_buffer.seek(0)
    file_bytes = np.asarray(bytearray(image_buffer.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    emotion = "unknown"
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        emotion = predict_emotion(face_roi)
        emotion_count[emotion] += 1
        break  # assuming only one face; adjust if needed
    
    # Store the image, prediction, and timestamp
    timestamp = str(datetime.now().time())  # Current time
    if data.get('store', False):  # Only store if the 'store' flag is true
        saved_frames.append({"timestamp": data['timestamp'], "image": encoded_image, "prediction": emotion})

    return jsonify({"emotion": emotion, "emotion_count": emotion_count, "timestamp": timestamp})

def gen():
    """Generate frame for video streaming."""
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            emotion = predict_emotion(face_roi)
            cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/report')
def report():
    global saved_frames
    report_frames = saved_frames.copy()  # Temporary store the frames for the report
    saved_frames.clear()  # Clear the saved frames
    return render_template('report.html', frames=report_frames)



if __name__ == '__main__':
    app.run(debug=True)

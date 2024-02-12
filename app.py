from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from keras.models import model_from_json
import os
import base64
import io
from datetime import datetime
from fpdf import FPDF
from flask import send_from_directory
from flask import make_response
import sqlite3
from collections import Counter

app = Flask(__name__)

# Create a connection to the database (or create a new one if it doesn't exist)
conn = sqlite3.connect("mydatabase.db")

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Define the table schema and create the table
cursor.execute('''
CREATE TABLE IF NOT EXISTS daily_visitors (
    name TEXT,
    emotion TEXT
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

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

def generate_emotion_report():

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Emotion Detection Report", ln=True, align='C')
    for entry in saved_frames:
        pdf.ln(10)
        pdf.cell(200, 10, txt="Timestamp: {} Emotion: {}".format(entry['timestamp'], entry['prediction']), ln=True, align='L')
    pdf_name = "emotion_report.pdf"
    pdf.output(pdf_name)
    return pdf_name


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
    return render_template('report.html', frames=saved_frames)

@app.route('/get_emotion_report')
def get_emotion_report():
    pdf_name = generate_emotion_report()
    return send_from_directory(os.getcwd(), pdf_name, as_attachment=True)

@app.route('/clear_data', methods=['POST'])
def clear_data():
    global saved_frames
    saved_frames.clear()
    return jsonify({"status": "Data cleared"})

@app.route('/start_new_session', methods=['POST'])
def start_new_session():
    global saved_frames, emotion_count
    saved_frames.clear()
    emotion_count = {label: 0 for label in emotion_labels}
    return jsonify({"status": "New session started"})

@app.route('/stop_video', methods=['POST'])
def stop_video():
    data = request.json
    name = data['name']
    emotion = data['emotion']
    
    # Insert data into SQLite database
    conn = sqlite3.connect("mydatabase.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO daily_visitors (name, emotion) VALUES (?, ?)", (name, emotion))
    conn.commit()
    conn.close()

    return jsonify({"status": "Data inserted successfully"})

@app.route('/get_daily_visitors', methods=['GET'])
def get_daily_visitors():
    # Create a connection to the database
    conn = sqlite3.connect("mydatabase.db")
    cursor = conn.cursor()

    # Execute a SELECT query
    cursor.execute("SELECT * FROM daily_visitors")

    # Fetch and store the results
    result = cursor.fetchall()

    # Close the connection
    conn.close()

    # Convert the result to a JSON response and return it
    visitors_data = [{"name": row[0], "emotion": row[1]} for row in result]
    
    # Count the total number of rows
    total_rows = len(visitors_data)

    # Extract emotions from the visitors data
    emotions = [visitor["emotion"] for visitor in visitors_data]

    # Count the occurrences of each emotion
    emotion_counts = Counter(emotions)

    # Find the most common emotion
    most_common_emotion = emotion_counts.most_common(1)

    # Create an object with count of rows and most common emotion
    result = [{
    "visitors": total_rows,
    "emotion": most_common_emotion[0][0] if most_common_emotion else None
    }]
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

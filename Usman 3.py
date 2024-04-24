from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
from deepface import DeepFace
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import os

app = Flask(__name__)

# Define STATIC_FOLDER to store static files
STATIC_FOLDER = r'P:\Working Directories\Flask\EmoEngage_3.9\static'
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Initialize OpenCV cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize global variables for emotion data and start time
emotion_data = {'time': [], 'angry': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}
start_time = time.time()

# Function to calculate attention score
def calculate_attention_score():
    total_weight = 0
    for emotion, weight_list in emotion_data.items():
        if emotion != 'time':
            total_weight += sum(weight_list)
    return total_weight

def detect_emotion():
    cap = cv2.VideoCapture(0)
    emotion_count = 0
    num_emotions = 100  # Define the number of emotions to capture
    emotion_interval = 1  # Define the interval between capturing emotions (in seconds)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            for result in results:
                emotion = result['dominant_emotion']
                print(f"Detected emotion: {emotion}")
                attention_weight = {
                    'angry': 0.5,
                    'disgust': 0.25,
                    'fear': 0.25,
                    'happy': 0.75,
                    'sad': 0.35,
                    'surprise': 1.0,
                    'neutral': 0.15,
                }.get(emotion, 0)
                current_time = time.time() - start_time
                emotion_data['time'].append(current_time)
                for key in emotion_data.keys():
                    if key != 'time':
                        if key == emotion:
                            emotion_data[key].append(attention_weight)
                        else:
                            emotion_data[key].append(0)
                emotion_count += 1

        if emotion_count >= num_emotions:
            total_weight = calculate_attention_score()
            print(f"Total Emotion Weight: {total_weight}")
            if total_weight > 1.5:
                print("Engaged")
            else:
                print("Not Engaged")
            break
        time.sleep(emotion_interval)

    print("Emotion data collected:")
    print(emotion_data)


# Route to display index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to get attention score
@app.route('/get_attention_score')
def get_attention_score():
    attention_score = calculate_attention_score()
    return jsonify({'attention_score': attention_score})

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to display emotion graph
@app.route('/emotion_graph')
def emotion_graph():
    plot_emotion_graph()
    return send_from_directory(app.config['STATIC_FOLDER'], 'emotion_graph.png')

# Function to plot emotion graph
def plot_emotion_graph():
    if not emotion_data['time']:
        print("No data available for plotting")
        return
    
    emotions = [emotion for emotion in emotion_data.keys() if emotion != 'time']
    total_emotion_weights = []
    for i, time_point in enumerate(emotion_data['time']):
        total_weight = sum([emotion_data[emotion][i] for emotion in emotions])
        total_emotion_weights.append(total_weight)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
    for i, emotion in enumerate(emotions):
        plt.bar(emotion_data['time'], [emotion_data[emotion][j] for j in range(len(emotion_data['time']))], width=5, color=colors[i], label=emotion)
    plt.xlabel('Time (s)')
    plt.ylabel('Emotion Weight')
    plt.title('Emotion Variation over Time')
    plt.xticks(emotion_data['time'], [f"{time_point:.0f}s" for time_point in emotion_data['time']])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'emotion_graph.png'))
    plt.close()

if __name__ == '__main__':
    # Start a separate thread for emotion detection
    t = threading.Thread(target=detect_emotion)
    t.start()
    
    # Run the Flask app
    app.run(debug=True)


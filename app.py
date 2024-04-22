import cv2
from flask import Flask, render_template, Response, send_from_directory, jsonify
from deepface import DeepFace
import matplotlib.pyplot as plt
import time
import os

app = Flask(__name__)
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.config['STATIC_FOLDER'] = STATIC_FOLDER

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_data = {'time': [], 'happiness': [], 'surprise': [], 'anger': [], 'disgust': [], 'fear': [], 'sadness': [], 'neutral': []}
start_time = time.time()

def calculate_attention_score():
    """Calculates the overall attention score based on emotion weights."""
    total_weight = sum(sum(weight_list) for emotion, weight_list in emotion_data.items() if emotion != 'time')
    attention_score = total_weight / len(emotion_data) if total_weight > 0 else 0
    return attention_score

def detect_emotion():
    """Captures video frames, detects faces, analyzes emotions, and updates emotion data."""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = frame[y:y + h, x:x + w]

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            except Exception as e:
                print(f"Error: Emotion analysis failed - {e}")
                continue

            if result and isinstance(result, list) and result:
                first_item = result[0]
                if isinstance(first_item, dict) and 'emotion' in first_item and 'emotion_score' in first_item:
                    emotion = first_item['emotion']
                    emotion_score = first_item['emotion_score']
                    print(f"Detected emotion: {emotion} with score: {emotion_score}")

                    attention_weights = {
                        'happiness': 1.0,
                        'surprise': 1.0,
                        'anger': 0.5,
                        'disgust': 0.5,
                        'fear': 0.5,
                        'sadness': 0.5,
                        'neutral': 0.0
                    }

                    emotion_data['time'].append(time.time() - start_time)
                    for emo in emotion_data.keys():
                        if emo != 'time':
                            emotion_data[emo].append(attention_weights.get(emo, 0))
                else:
                    print("Warning: Unexpected format of emotion result")

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Unable to encode frame")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_attention_score')
def get_attention_score():
    """Returns the calculated attention score in JSON format."""
    attention_score = calculate_attention_score()
    return jsonify({'attention_score': attention_score})


@app.route('/video_feed')
def video_feed():
    """Provides the video feed with error handling."""
    try:
        return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error serving video feed: {e}")
        return Response(status=500)


@app.route('/emotion_graph')
def emotion_graph():
    """Generates and serves the emotion graph as a PNG image."""
    if not emotion_data['time']:
        return "No data available for plotting"

    plt.switch_backend('Agg')  # Switch backend to Agg
    plt.figure(figsize=(10, 6))
    for emo in emotion_data.keys():
        if emo != 'time':
            plt.plot(emotion_data['time'], emotion_data[emo], label=emo)
    plt.xlabel('Time (s)')
    plt.ylabel('Emotion')
    plt.title('Emotion over Time')
    plt.legend()
    plt.grid(True)

    graph_path = os.path.join(app.config['STATIC_FOLDER'], 'emotion_graph.png')
    plt.savefig(graph_path)  # Save the graph as a PNG file
    plt.close()  # Close the plot to release resources

    return send_from_directory(app.config['STATIC_FOLDER'], 'emotion_graph.png')


if __name__ == '__main__':
    app.run(debug=True)

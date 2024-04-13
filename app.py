from flask import Flask, render_template, Response, send_from_directory, jsonify
import cv2
from deepface import DeepFace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

app = Flask(__name__)
STATIC_FOLDER = r'P:\Working Directories\Flask\EmoEngage_3.9\static'
app.config['STATIC_FOLDER'] = STATIC_FOLDER

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
url = 'http://192.168.18.19:4747'

emotion_data = {'time': [], 'anger': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}
start_time = time.time()

def calculate_attention_score():
  
  # Calculate attention score based on emotion data (modify as needed)
  total_weight = 0
  attention_score = 0
  
  # Print for debugging
  print(f"emotion_data.keys(): {emotion_data.keys()}")
  print(f"emotion_data.values(): {emotion_data.values()}")
  
  for emotion, weight_list in emotion_data.items():  
    if emotion != 'time':
        weight_list.append(total_weight)

  if total_weight > 0:
    attention_score /= total_weight  # Normalize by total weight (optional)
  return attention_score


def detect_emotion():
  cap = cv2.VideoCapture(0)
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
      result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
      if result and result['confidence'] > 0.7:  # Filter based on confidence score
        emotion = result[0]['emotion']
        print(f"Detected emotion: {emotion}")  # Track detected emotion
        attention_weight = {
            'happiness': 1.0,
            'surprise': 1.0,
            'anger': 0.5,  # Lower weight for negative emotions
            'frustration': 0.5,
            'disgust': 0.25,  # Even lower weight for strong negative emotions
            'fear': 0.25,
            'sadness': 0.25,
            'neutral': 0.0   # Neutral indicates low attention
        }.get(emotion, 0)  # Default weight for unknown emotions

        # Update emotion data with attention score
        emotion_data['time'].append(time.time() - start_time)
        emotion_data[emotion].append(attention_weight)

    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
      print("Error: Unable to encode frame")
      break

    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/get_attention_score')
def get_attention_score():
  attention_score = calculate_attention_score()
  return jsonify({'attention_score': attention_score})  # Return attention score as JSON

@app.route('/participants')
def participants():
    return render_template('page2.html')

@app.route('/details')
def details():
    return render_template('page3.html')


@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_graph')
def emotion_graph():
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

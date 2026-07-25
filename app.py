from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64

app = Flask(__name__)
model = load_model('emotion_recognition_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()
        image_data = data['image']
        image_data = image_data.split(",")[1]
        decoded_data = base64.b64decode(image_data)
        np_data = np.frombuffer(decoded_data, np.uint8)
        
        if np_data.size == 0:
            return jsonify({"error": "Empty buffer received"}), 400

        img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({"error": "Image decoding failed"}), 400

        img = cv2.resize(img, (48, 48))
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        prediction = model.predict(img)
        emotion_labels = ['Angry', 'Disgust', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Fear']
        emotion = emotion_labels[np.argmax(prediction)]

        return jsonify({"emotion": emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

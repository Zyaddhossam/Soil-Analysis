from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf  

app = Flask(__name__)

class_dict = {
    "Alluvial soil": 0,
    "Black Soil": 1,
    "Clay soil": 2,
    "Red soil": 3
}

model = tf.keras.models.load_model('model.h5')  

def predict_image(img):
    labels = list(class_dict.keys())
    resized_img = img.resize((299, 299))
    img_array = np.asarray(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)[0] 

    max_index = np.argmax(predictions)  
    predicted_label = labels[max_index]
    predicted_probability = predictions[max_index]

    return predicted_label, predicted_probability

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        img = Image.open(io.BytesIO(file.read()))
        predicted_label, predicted_probability = predict_image(img)

        response = {
            'predicted_label': predicted_label,
            'predicted_probability': round(float(predicted_probability), 2)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

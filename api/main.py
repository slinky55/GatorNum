from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

#model = tf.keras.models.load_model('path_to_your_model')

# Define a route to handle the POST request with an image
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure that a file named 'image' is sent in the POST request
    request_json = request.get_json()
    
    if 'image' not in request_json:
        return jsonify({'error': 'No image provided'})

    # Get the image file from the POST request
    image_file = request_json['image']
    print(image_file)

    # Make prediction using the loaded TensorFlow model
    # prediction = model.predict(np.array([image]))
    output_label = ["this is a test"]

    # Send the output back to the frontend
    return jsonify({'prediction': output_label})


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=5000, debug=True) this is the default and runs localhost 5000
    app.run(debug=True)

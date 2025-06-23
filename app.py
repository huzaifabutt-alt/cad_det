import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
import numpy as np
app = Flask(__name__)
model_path = 'model1.h5'
model=load_model(model_path)
img_size = (128, 128)  

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    # Read and preprocess image
    img_bytes = file.read()
    img = load_img(BytesIO(img_bytes), target_size=img_size)
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array_expanded)
    predicted_class = int(prediction[0][0] > 0.5)
    inv_label_map = {0: 'normal', 1: 'sick'}
    class_label = inv_label_map[predicted_class]

    print("Prediction raw value:", prediction[0][0])
    print("Predicted class index:", predicted_class)
    print("Predicted label:", class_label)

    # Create directories if they don't exist
    save_dir = os.path.join(os.getcwd(), class_label)
    os.makedirs(save_dir, exist_ok=True)

    # Save the original image to the appropriate directory
    filename = file.filename or 'uploaded_image.jpg'
    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'wb') as f:
        f.write(img_bytes)

    # Redirect based on prediction
    if class_label == 'normal':
        return redirect(url_for('result_normal'))
    else:
        return redirect(url_for('result_sick'))

@app.route('/result/normal')
def result_normal():
    return render_template('result_normal.html')


@app.route('/result/sick')
def result_sick():
    return render_template('result_sick.html')


if __name__ == '__main__':
    app.run(debug=True)

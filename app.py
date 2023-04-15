import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import io
import base64
from PIL import Image
from keras.preprocessing.image import image_utils
from keras.models import load_model


# Replace this path with the path to your saved model
MODEL_PATH = 'trained_models/my_model.h5'

# Load the saved model
model = load_model(MODEL_PATH)


app = Flask(__name__)


# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('upload_image'))
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image and classify it using the loaded model
        image = Image.open(file_path)
        image = image.resize((128, 128))  # Resize the image to the expected input size
        image_array = image_utils.img_to_array(image) / 255.0  # Convert the image to an array and normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add a batch dimension

        prediction = model.predict(image_array)  # Classify the image
        classification = 'Dog' if prediction[0][0] < 0.5 else 'Cat'  # Determine the classification

        # Encode the image to base64 for displaying it in the HTML template
        image_data = io.BytesIO()
        image.save(image_data, "JPEG")
        image_base64 = base64.b64encode(image_data.getvalue()).decode('ascii')

        return render_template('result.html', classification=classification, image_base64=image_base64)

    return 'Invalid file type', 400

if __name__ == '__main__':
    app.run(debug=True)

import os
from flask import Flask, request, render_template
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
MODEL_PATH = "models/micro_doppler_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

# Ensure directories exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ========== Spectrogram generation and classification ==========

def create_spectrogram(image_name, frequency_data, title):
    plt.figure(figsize=(10, 5))
    plt.imshow(frequency_data, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title(title)
    plt.xlabel('Time Bins')
    plt.ylabel('Frequency Bins')
    plt.savefig(image_name)
    plt.close()

def generate_spectrogram_data(amplitudes):
    time_bins = 100
    frequency_bins = len(amplitudes)  # Number of frequencies corresponds to input amplitudes
    freq_data = np.zeros((frequency_bins, time_bins))

    for f in range(frequency_bins):
        frequency = 5 + (25 / (frequency_bins - 1)) * f  # Linear spacing from 5 to 30
        amplitude = amplitudes[f]
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * np.linspace(0, 1, time_bins))
        freq_data[f] = sine_wave

    image_name = 'static/drone_spectrogram.png'  # Output image path
    title = 'Synthetic Spectrogram'
    create_spectrogram(image_name, freq_data, title)

    return image_name, freq_data

def classify_spectrogram(frequency_data):
    average_intensity = np.mean(frequency_data)
    max_intensity = np.max(frequency_data)
    
    if average_intensity < 0.5 and max_intensity < 0.75:
        return "Bird"
    else:
        return "Drone"

# ========== Routes ==========

@app.route('/')
def index():
    return render_template('index.html')

# Handle amplitude form submission (generate spectrogram + classify)
@app.route('/generate', methods=['POST'])
def generate():
    amplitudes_input = request.form.get('amplitudes')
    if not amplitudes_input:
        return "Please provide amplitude inputs."

    try:
        amplitudes = list(map(float, amplitudes_input.split(',')))
    except Exception as e:
        return f"Invalid amplitude input format: {e}"

    image_path, frequency_data = generate_spectrogram_data(amplitudes)
    classification = classify_spectrogram(frequency_data)

    return render_template('result.html', 
                           image_path=image_path, 
                           classification=classification, 
                           prediction=None, 
                           file_path=None)

# Handle image upload form submission (classify uploaded image)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Preprocess and predict
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_label = 'Drone' if prediction[0][0] > 0.5 else 'Bird'

    return render_template('result.html', 
                           image_path=None, 
                           classification=None, 
                           prediction=class_label, 
                           file_path='uploads/' + file.filename)

# ========== Run app ==========

if __name__ == '__main__':
    app.run(debug=True)

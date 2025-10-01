import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import os

# Define the model directory
model_dir = "models"  # or use the absolute path as explained above

# Check if the directory exists; if not, create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Now you can save your model here
MODEL_SAVE_PATH = os.path.join(model_dir, "micro_doppler_model.h5")


# Dataset directories
BIRD_DIR = r"C:\Users\lakks\Downloads\archive\BirdVsDrone\Birds"
DRONE_DIR = r"C:\Users\lakks\Downloads\archive\BirdVsDrone\Drones"
MODEL_SAVE_PATH = "models/micro_doppler_model.h5"

# Image size
IMAGE_SIZE = (128, 128)

# Create the data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    directory=r"C:\Users\lakks\Downloads\archive\BirdVsDrone",
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='binary',
    subset='training')

val_generator = datagen.flow_from_directory(
    directory=r"C:\Users\lakks\Downloads\archive\BirdVsDrone",
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='binary',
    subset='validation')

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: Bird or Drone
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)

print("Model trained and saved!")

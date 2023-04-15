import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
from keras.callbacks import ReduceLROnPlateau


use_prepared_pic = False
cat_folder = './dogs_and_cats/cats/'
dog_folder = './dogs_and_cats/dogs/'
pic_target_size = (128, 128)
data_validation_split=0.2
seed_number = 42
set_lr = 0.001
input_shape = (128, 128, 3) # Define the input shape based on the preprocessed images
epochs = 15

def load_images_from_folder(folder, target_size=pic_target_size):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            img_resized = img.resize(target_size, Image.ANTIALIAS)
            images.append(np.array(img_resized))
    return images

def preprocess_images(images, normalize=True):
    if normalize:
        return np.array(images) / 255.0
    else:
        return np.array(images)

def prepare_dataset(cat_folder, dog_folder, target_size=pic_target_size, validation_split=data_validation_split):
    # Load and preprocess cat images
    cat_images = load_images_from_folder(cat_folder, target_size)
    cat_images = preprocess_images(cat_images)
    cat_labels = np.ones(len(cat_images))

    # Load and preprocess dog images
    dog_images = load_images_from_folder(dog_folder, target_size)
    dog_images = preprocess_images(dog_images)
    dog_labels = np.zeros(len(dog_images))

    # Combine cat and dog images and labels
    images = np.concatenate([cat_images, dog_images], axis=0)
    labels = np.concatenate([cat_labels, dog_labels], axis=0)

    # Shuffle the dataset
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)

    # Split the dataset into training and validation sets
    split_idx = int(len(images) * (1 - validation_split))
    train_images, train_labels = images[:split_idx], labels[:split_idx]
    val_images, val_labels = images[split_idx:], labels[split_idx:]

    return (np.array(train_images), np.array(train_labels)), (np.array(val_images), np.array(val_labels))


if 'use_prepared_pic' in locals() or 'use_prepared_pic' in globals():
    if use_prepared_pic:
        train_images_path = '.prepared_pic_arrays/train_images.npy'
        train_labels_path = '.prepared_pic_arrays/train_labels.npy'
        val_images_path = '.prepared_pic_arrays/val_images.npy'
        val_labels_path = '.prepared_pic_arrays/val_labels.npy'

        if os.path.exists(train_images_path) and os.path.exists(train_labels_path) and os.path.exists(val_images_path) and os.path.exists(val_labels_path):
            train_images = np.load(train_images_path)
            train_labels = np.load(train_labels_path)
            val_images = np.load(val_images_path)
            val_labels = np.load(val_labels_path)
            print("Loaded prepared data.")
        else:
            print("Prepared data not found. Please check the file paths.")
    else:
        print("use_prepared_pic is set to False. Preparing data and saving to: ./prepared_pic_arrays/")
        (train_images, train_labels), (val_images, val_labels) = prepare_dataset(cat_folder, dog_folder)
        np.save('prepared_pic_arrays/train_images.npy', train_images)
        np.save('prepared_pic_arrays/train_labels.npy', train_labels)
        np.save('prepared_pic_arrays/val_images.npy', val_images)
        np.save('prepared_pic_arrays/val_labels.npy', val_labels)
else:
    print("use_prepared_pic is not defined.")


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

set_seed(seed_number)  # Replace 42 with your desired seed value

# Create a simple neural network model
def create_model(input_shape):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(0.3),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=set_lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model



model = create_model(input_shape)

# Train the model

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

history = model.fit(train_images, train_labels,
                    epochs=epochs,
                    validation_data=(val_images, val_labels),
                    verbose=0,  # Set verbose to 0 to disable the default progress bar
                    callbacks=[TqdmCallback(verbose=1), reduce_lr])  # Add the custom TqdmCallback to display the progress bar
"""
history = model.fit(train_images, train_labels,
                    epochs=epochs,
                    validation_data=(val_images, val_labels),
                    verbose=0,  # Set verbose to 0 to disable the default progress bar
                    callbacks=[TqdmCallback(verbose=1)])  # Add the custom TqdmCallback to display the progress bar
"""

"""
history = model.fit(train_images, train_labels,
                    epochs=epochs,
                    validation_data=(val_images, val_labels))
"""



# Plot the training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history)

model.save('trained_models/my_model.h5')


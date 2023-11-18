import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import zipfile

from tf_keras import models
from tf_keras import layers

from tf_keras.callbacks import ModelCheckpoint, EarlyStopping
from tf_keras.utils import to_categorical

from keras_preprocessing.image import ImageDataGenerator


def draw_number(pixels):
    plt.figure()
    plt.imshow(pixels)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def load_emnist(key_path):
    # Check if the Kaggle API key file exists in the specified location
    if not os.path.isfile(os.path.expanduser("~/.kaggle/kaggle.json")):
        # Create the ~/.kaggle directory if it doesn't exist
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        # Copy the Kaggle API key from the provided path
        os.system(f'cp {key_path} ~/.kaggle/kaggle.json')
        # Set the appropriate permissions for the API key file
        os.system('chmod 600 ~/.kaggle/kaggle.json')
        # Install the Kaggle Python package if not already installed
        os.system('pip install kaggle')

    # Download the EMNIST dataset from Kaggle
    if not os.path.isfile("data/emnist.zip"):
        os.system('kaggle datasets download -d crawford/emnist -p data')

    # Extract the downloaded zip file
    if not os.path.exists("data/emnist"):
        with zipfile.ZipFile("data/emnist.zip", 'r') as zip_ref:
            zip_ref.extractall("data/emnist")


def pad_images(images, target_size=(32, 32)):
    padded_images = np.zeros((images.shape[0], target_size[0], target_size[1], 1))
    padding = [(target_size[i] - images.shape[i + 1]) // 2 for i in range(2)]

    padded_images[:, padding[0]:padding[0] + images.shape[1], padding[1]:padding[1] + images.shape[2], :] = images
    return padded_images


load_emnist("kaggle.json")

# Load the data
train = pd.read_csv('data/emnist/emnist-byclass-train.csv')
test = pd.read_csv('data/emnist/emnist-byclass-test.csv')

# Preprocess the data
train_labels = train.iloc[:, 0]
train_images = train.iloc[:, 1:]
test_labels = test.iloc[:, 0]
test_images = test.iloc[:, 1:]

train_images = train_images.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_images = test_images.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

train_images = pad_images(train_images)
test_images = pad_images(test_images)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=[0.90, 1.1],  # This zooms both in and out
    shear_range=0.1,
    fill_mode='nearest'
)
datagen.fit(train_images)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(35, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(train_images, train_labels, batch_size=128),
                    epochs=50,
                    validation_data=(test_images, test_labels),
                    steps_per_epoch=len(train_images) / 128,
                    callbacks=[early_stopping, checkpoint])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

import os
import subprocess
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.preprocessing import LabelBinarizer

from tf_keras import models
from tf_keras import layers

from tf_keras.callbacks import ModelCheckpoint, EarlyStopping


def draw_number(pixels):
    plt.figure()
    plt.imshow(pixels)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def load_emnist(key_path):
    # Check if the Kaggle API key file exists
    kaggle_key_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.isfile(kaggle_key_path):
        # Create the .kaggle directory if it doesn't exist
        os.makedirs(os.path.dirname(kaggle_key_path), exist_ok=True)

        # Move the Kaggle API key to the correct location
        subprocess.run(['cp', key_path, '~/.kaggle'], check=True)

        # Set the correct permissions for the API key
        os.chmod(kaggle_key_path, 0o600)

        # Install the Kaggle package
        subprocess.run(['pip', 'install', 'kaggle'], check=True)

        # Download the dataset using Kaggle API
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 'crawford/emnist', '-p', 'data'], check=True)

        # Unzip the downloaded file
        with zipfile.ZipFile("data/emnist.zip", 'r') as zip_ref:
            zip_ref.extractall("data/emnist")


def resize_images(images):
    res = []
    for img in images:
        r = img.reshape(28, 28).astype('uint8')

        b = Image.fromarray(r)
        f = b.resize((32, 32))

        res.append(np.array(f).astype('float32') / 255.0)
    return np.array(res)


load_emnist("kaggle.json")

train = pd.read_csv("data/emnist/emnist-byclass-train.csv").to_numpy()
test = pd.read_csv("data/emnist/emnist-byclass-test.csv").to_numpy()

# Preprocess the data
train_labels = train[:, 0]
train_images = train[:, 1:]
test_labels = test[:, 0]
test_images = test[:, 1:]

train_images = resize_images(train_images)
test_images = resize_images(test_images)

print(train_images.shape)
print(train_images[0])

LB = LabelBinarizer()
train_labels = np.array(LB.fit_transform(train_labels))
test_labels = np.array(LB.fit_transform(test_labels))
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)

print(train_labels.shape)
print(train_labels)

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
model.add(layers.Dense(62, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels,
                    epochs=50,
                    batch_size=32,
                    validation_data=(test_images, test_labels),
                    callbacks=[early_stopping, checkpoint])

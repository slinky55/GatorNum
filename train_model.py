from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tf_keras.src.datasets import mnist
from tf_keras.src import backend, callbacks
from tf_keras.src.optimizers import SGD
from tf_keras.src.preprocessing.image import ImageDataGenerator
from tf_keras.src.regularizers import l2
from tf_keras.src.layers import Activation
from tf_keras.src.layers import Flatten, Dense, BatchNormalization, AveragePooling2D, Conv2D, add
from tf_keras import Model, Input

import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_number(pixels):
    plt.figure()
    plt.imshow(pixels)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def load_letters():
    data = []
    letter_labels = []

    for row in open("data/emnist/emnist-letters-train.csv"):
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(i) for i in row[1:]], dtype="uint8")

        image = image.reshape((28, 28))

        image = cv2.flip(image, 1)

        # Rotate the image 90 degrees anti-clockwise
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        data.append(image)
        letter_labels.append(label)

    for row in open("data/emnist/emnist-letters-test.csv"):
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(i) for i in row[1:]], dtype="uint8")

        image = image.reshape((28, 28))

        image = cv2.flip(image, 1)

        # Rotate the image 90 degrees anti-clockwise
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        data.append(image)
        letter_labels.append(label)

    data = np.array(data, dtype="float32")
    letter_labels = np.array(letter_labels, dtype="int")

    return data, letter_labels


def load_numbers():
    (trainData, trainLabels), (testData, testLabels) = mnist.load_data()
    data = np.vstack([trainData, testData])
    num_labels = np.hstack([trainLabels, testLabels])

    return data, num_labels


img_shape = (32, 32, 1)
chanDim = -1

# hyperparameters
eps = 2e-5
reg = 0.0005
mom = 0.9
lr = 1e-1
batch_size = 128
epochs = 50


def get_residual(d, k, s, r=False):
    short = d

    # First block of ResNet (1x1)
    bn1 = BatchNormalization(axis=chanDim, epsilon=eps,
                             momentum=mom)(d)
    act1 = Activation("relu")(bn1)
    conv1 = Conv2D(int(k * 0.25), (1, 1), use_bias=False,
                   kernel_regularizer=l2(reg))(act1)

    # Second block of ResNet (3x3)
    bn2 = BatchNormalization(axis=chanDim, epsilon=eps,
                             momentum=mom)(conv1)
    act2 = Activation("relu")(bn2)
    conv2 = Conv2D(int(k * 0.25), (3, 3), strides=s,
                   padding="same", use_bias=False,
                   kernel_regularizer=l2(reg))(act2)

    # Third block of ResNet (1x1)
    bn3 = BatchNormalization(axis=chanDim, epsilon=eps,
                             momentum=mom)(conv2)
    act3 = Activation("relu")(bn3)
    conv3 = Conv2D(k, (1, 1), use_bias=False,
                   kernel_regularizer=l2(reg))(act3)

    if r:
        short = Conv2D(k, (1, 1), strides=s,
                       use_bias=False, kernel_regularizer=l2(reg))(act1)

    out = add([conv3, short])

    return out


img_shape = (32, 32, 1)
chanDim = -1

# hyperparameters
eps = 2e-5
reg = 0.0005
mom = 0.9
lr = 1e-1
batch_size = 128
epochs = 50


def get_residual(d, k, s, r=False):
    short = d

    # First block of ResNet (1x1)
    bn1 = BatchNormalization(axis=chanDim, epsilon=eps,
                             momentum=mom)(d)
    act1 = Activation("relu")(bn1)
    conv1 = Conv2D(int(k * 0.25), (1, 1), use_bias=False,
                   kernel_regularizer=l2(reg))(act1)

    # Second block of ResNet (3x3)
    bn2 = BatchNormalization(axis=chanDim, epsilon=eps,
                             momentum=mom)(conv1)
    act2 = Activation("relu")(bn2)
    conv2 = Conv2D(int(k * 0.25), (3, 3), strides=s,
                   padding="same", use_bias=False,
                   kernel_regularizer=l2(reg))(act2)

    # Third block of ResNet (1x1)
    bn3 = BatchNormalization(axis=chanDim, epsilon=eps,
                             momentum=mom)(conv2)
    act3 = Activation("relu")(bn3)
    conv3 = Conv2D(k, (1, 1), use_bias=False,
                   kernel_regularizer=l2(reg))(act3)

    if r:
        short = Conv2D(k, (1, 1), strides=s,
                       use_bias=False, kernel_regularizer=l2(reg))(act1)

    out = add([conv3, short])

    return out


(letters_images, letters_labels) = load_letters()
(digits_images, digits_labels) = load_numbers()

# Letter labels will start at 10, numbers are 0-9
letters_labels += 10

images = np.vstack([letters_images, digits_images])
labels = np.hstack([letters_labels, digits_labels])

# resize to fit ResNet architecture
images = [cv2.resize(i, (32, 32)) for i in images]
images = np.array(images, dtype="float32")

images = np.expand_dims(images, axis=-1)
images /= 255.0  # normalize

LB = LabelBinarizer()
labels = LB.fit_transform(labels)

totals = labels.sum(axis=0)
weights = {}

# loop over all classes and calculate the class weight
for i in range(0, len(totals)):
    weights[i] = totals.max() / totals[i]

(train_images, test_images, train_labels, test_labels) = train_test_split(images,
                                                                          labels, test_size=0.20, stratify=labels,
                                                                          random_state=42)

dataGen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest")

stages = (3, 3, 3)
filters = (64, 64, 128, 256)

if backend.image_data_format() == "channels_first":
    img_shape = (1, 32, 32)
    chanDim = 1

inputs = Input(shape=img_shape)
x = BatchNormalization(axis=-1, epsilon=eps,
                       momentum=mom)(inputs)
x = Conv2D(filters[0], (3, 3), use_bias=False,
           padding="same", kernel_regularizer=l2(reg))(x)

for i in range(0, len(stages)):
    stride = (1, 1) if i == 0 else (2, 2)
    x = get_residual(x, filters[i + 1], stride, r=True)

    for j in range(0, stages[i] - 1):
        x = get_residual(x, filters[i + 1], (1, 1))

x = BatchNormalization(axis=chanDim, epsilon=eps,
                       momentum=mom)(x)
x = Activation("relu")(x)
x = AveragePooling2D((8, 8))(x)

x = Flatten()(x)
x = Dense(len(LB.classes_), kernel_regularizer=l2(reg))(x)
x = Activation("softmax")(x)

# create the model
model = Model(inputs, x, name="ResNet")
opt = SGD(learning_rate=lr, decay=lr / epochs)

model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath='model',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1)

# Callback for early stopping
early_stopping_callback = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,
    mode='min',
    restore_best_weights=True)

history = model.fit(
    dataGen.flow(train_images, train_labels, batch_size=batch_size),
    validation_data=(test_images, test_labels),
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
    batch_size=batch_size,
    class_weight=weights,
    callbacks=[model_checkpoint_callback, early_stopping_callback],
    verbose=1)
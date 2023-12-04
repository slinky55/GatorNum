from tf_keras import backend, callbacks
from tf_keras.src.optimizers import SGD
from tf_keras.src.preprocessing.image import ImageDataGenerator
from tf_keras.src.regularizers import l2
from tf_keras.src.layers import Activation
from tf_keras.src.layers import Flatten, Dense, BatchNormalization, AveragePooling2D, Conv2D, add
from tf_keras import Model, Input

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from tf_keras.src.datasets import mnist
import cv2
import matplotlib.pyplot as plt

def draw_number(pixels):
    plt.figure()
    plt.imshow(pixels)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def load_numbers():
    (trainData, trainLabels), (testData, testLabels) = mnist.load_data()
    data = np.vstack([trainData, testData])
    num_labels = np.hstack([trainLabels, testLabels])

    return data, num_labels

def load_letters():
    letters = []
    l_labels = []

    for row in open("data/a-z.csv"):
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")

        image = image.reshape((28, 28))

        letters.append(image)
        l_labels.append(label)

    letters = np.array(letters, dtype="float32")
    l_labels = np.array(l_labels, dtype="int")

    return letters, l_labels

chanDim = -1

def get_res(data, K, stride, chanDim, red=False,
            reg=0.0001, eps=2e-5, mom=0.9):
    shortcut = data

    # 1x1 Conv Filter
    bn1 = BatchNormalization(axis=chanDim, epsilon=eps,
        momentum=mom)(data)
    act1 = Activation("relu")(bn1)
    conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
        kernel_regularizer=l2(reg))(act1)

    # 3x3 Conv Filter
    bn2 = BatchNormalization(axis=chanDim, epsilon=eps,
        momentum=mom)(conv1)
    act2 = Activation("relu")(bn2)
    conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
        padding="same", use_bias=False,
        kernel_regularizer=l2(reg))(act2)

    # 1x1 Conv Filter
    bn3 = BatchNormalization(axis=chanDim, epsilon=eps,
        momentum=mom)(conv2)
    act3 = Activation("relu")(bn3)
    conv3 = Conv2D(K, (1, 1), use_bias=False,
        kernel_regularizer=l2(reg))(act3)

    if red:
        shortcut = Conv2D(K, (1, 1), strides=stride,
            use_bias=False, kernel_regularizer=l2(reg))(act1)

    x = add([conv3, shortcut])

    return x

def get_model(width, height, depth, classes, stages, filters,
		reg=0.0001, eps=2e-5, mom=0.9):

		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if backend.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# set the input and then apply a BN followed by CONV
		inputs = Input(shape=inputShape)
		x = BatchNormalization(axis=chanDim, epsilon=eps,
			momentum=mom)(inputs)
		x = Conv2D(filters[0], (3, 3), use_bias=False,
			padding="same", kernel_regularizer=l2(reg))(x)

		# loop over the number of stages
		for i in range(0, len(stages)):
			# initialize the stride, then apply a residual module
			# used to reduce the spatial size of the input volume
			stride = (1, 1) if i == 0 else (2, 2)
			x = get_res(x, filters[i + 1], stride,
                        chanDim, red=True, bnEps=eps, bnMom=mom)

			# loop over the number of layers in the stage
			for j in range(0, stages[i] - 1):
				# apply a ResNet module
				x = get_res(x, filters[i + 1],
                            (1, 1), chanDim, bnEps=eps, bnMom=mom)

		# apply BN => ACT => POOL
		x = BatchNormalization(axis=chanDim, epsilon=eps,
			momentum=mom)(x)
		x = Activation("relu")(x)
		x = AveragePooling2D((8, 8))(x)

		# softmax classifier
		x = Flatten()(x)
		x = Dense(classes, kernel_regularizer=l2(reg))(x)
		x = Activation("softmax")(x)

		# create the model
		model = Model(inputs, x, name="GatorNet")

		# return the constructed network architecture
		return model

img_shape = (32, 32, 1)

stages = (3, 3, 3)
filters = (64, 64, 128, 256)

# hyperparameters
lr = 1e-1
batch_size = 128
epochs = 50

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


model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath='model',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1)

# Callback for early stopping
early_stopping_callback = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,
    mode='min',
    restore_best_weights=True)

model = get_model(32, 32, 1, len(LB.classes_), stages,
	filters, reg=0.0005)

model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=lr, decay=lr / epochs),
	metrics=["accuracy"])

history = model.fit(
    dataGen.flow(train_images, train_labels, batch_size=batch_size),
    validation_data=(test_images, test_labels),
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
    class_weight=weights,
    callbacks=[model_checkpoint_callback, early_stopping_callback],
    verbose=1)
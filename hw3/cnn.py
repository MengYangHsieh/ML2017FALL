#!/usr/bin/env python3

from utils import extract_training_data

import numpy as np
import pickle
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

training_file_name = sys.argv[1]
model_file_name = sys.argv[2]

num_classes = 7

training_x, training_y = extract_training_data(training_file_name)
training_y = to_categorical(training_y, num_classes=num_classes)

training_x = np.array(
    training_x, dtype=np.float64).reshape((len(training_x), 48, 48, 1))
training_y = np.array(training_y, dtype=np.float64)

training_x /= 255

shuffle_index = np.arange(len(training_x))
np.random.seed(19940622)
np.random.shuffle(shuffle_index)
training_x = training_x[shuffle_index]
training_y = training_y[shuffle_index]

num_validating_x = training_x.shape[0] // 10
validating_x = training_x[:num_validating_x]
training_x = training_x[num_validating_x:]
validating_y = training_y[:num_validating_x]
training_y = training_y[num_validating_x:]

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1)))
model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

batch_size = 256
history = model.fit_generator(
    train_datagen.flow(training_x, training_y, batch_size=batch_size),
    steps_per_epoch=100,
    validation_data=(validating_x, validating_y),
    epochs=15,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3)])

model.save(model_file_name)

if len(sys.argv) > 3:
    dump_file_name = sys.argv[3]
    with open(dump_file_name, "wb") as dump_file:
        pickle.dump({"history": history.history}, dump_file)


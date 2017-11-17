#!/usr/bin/env python3

from utils import *

import numpy as np
import pickle
import sys

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

training_file_name = sys.argv[1]
testing_file_name = sys.argv[2]
model_file_name = sys.argv[3]

num_classes = 7

training_x, training_y = extract_training_data(training_file_name)
training_y = to_categorical(training_y, num_classes=num_classes)
testing_x = extract_testing_data(testing_file_name)

training_x = np.array(
    training_x, dtype=np.float64).reshape((len(training_x), 48, 48, 1))
training_y = np.array(training_y, dtype=np.float64)
testing_x = np.array(
    testing_x, dtype=np.float64).reshape((len(testing_x), 48, 48, 1))

training_x /= 255
testing_x /= 255

shuffle_index = np.arange(len(training_x))
np.random.seed(19930408)
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

num_testing_xs = testing_x.shape[0]

model = load_model(model_file_name)
model.summary()

history = {"acc": [], "val_acc": [], "loss": [], "val_loss": []}

batch_size = 256

num_new_training_xs = 0
nth_iteration = 0
while num_new_training_xs < num_testing_xs * 0.7:
    nth_iteration += 1

    predicted_y = model.predict(testing_x)

    thresh = 0.8
    for i, ys in enumerate(predicted_y):
        predicted_classes = list(map(lambda y: y >= thresh, ys))
        if any(predicted_classes):
            training_x = np.concatenate((training_x, [testing_x[i]]))
            training_y = np.concatenate(
                (training_y,
                 [list(map(lambda y: 1 if y else 0, predicted_classes))]))
            num_new_training_xs += 1

    print("{:d} nth self-training, {:d} test labels".format(
        nth_iteration, num_new_training_xs))

    tmp_history = model.fit_generator(
        train_datagen.flow(training_x, training_y, batch_size=batch_size),
        steps_per_epoch=100,
        validation_data=(validating_x, validating_y),
        epochs=15,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2)])

    history["acc"] += tmp_history.history["acc"]
    history["val_acc"] += tmp_history.history["val_acc"]
    history["loss"] += tmp_history.history["loss"]
    history["val_loss"] += tmp_history.history["val_loss"]

model.save(model_file_name)

if len(sys.argv) > 4:
    dump_file_name = sys.argv[4]
    with open(dump_file_name, "wb") as dump_file:
        pickle.dump({"history": history}, dump_file)


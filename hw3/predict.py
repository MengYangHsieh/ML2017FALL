#!/usr/bin/env python3

from utils import extract_testing_data

import numpy as np
import pickle
import sys

from keras.models import load_model, Sequential

testing_file_name = sys.argv[1]
output_file_name = sys.argv[2]
model_file_name = sys.argv[3]

testing_x = extract_testing_data(testing_file_name)
testing_x = np.array(
    testing_x, dtype=np.float64).reshape((len(testing_x), 48, 48, 1))

testing_x /= 255

model = load_model(model_file_name)
model.summary()

testing_y = model.predict_classes(testing_x)

with open(output_file_name, "w") as output_file:
    output_file.write("id,label\n")
    for i, n in enumerate(testing_y):
        output_file.write("%d,%d\n" % (i, n))

# batch_size = 20
# loss_and_metrics = model.evaluate(testing_x, testing_y, batch_size)
# val_proba = model.predict(testing_x)
# val_classes = val_proba.argmax(axis=-1)

# with open(output_file_name, 'w') as f:
    # f.write('acc = %s\n' % str(loss_and_metrics[1]))
    # f.write('id,label')
    # for i in range(len(val_classes)):
        # f.write('\n' + str(i) + ',' + str(val_classes[i]))


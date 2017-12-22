#!/usr/bin/env python3

import mf
import dnn

from config import METHOD
from config import SEED, NUM_MODELS, VALIDATION_RATIO
from config import NUM_USERS, NUM_MOVIES, LATENT_DIMENSION
from config import IS_NORMALIZED, IS_REGULARIZED, LAMBDA, IS_BIASED
from extract import extract_x_test, extract_users, extract_movies

import numpy as np

import os
import sys
import pickle

testing_file_name = sys.argv[1]
output_file_name = sys.argv[2]
movies = sys.argv[3]
users = sys.argv[4]
suffix = sys.argv[5]


if METHOD == "DNN":
    IS_BIASED = False

x_test, x_test_ids = extract_x_test(testing_file_name, is_biased=IS_BIASED)

if METHOD == "MF":
    model = mf.build(
        num_users=NUM_USERS,
        num_movies=NUM_MOVIES,
        latent_dimension=LATENT_DIMENSION,
        is_regularized=IS_REGULARIZED,
        lamda=LAMBDA,
        is_biased=IS_BIASED)
if METHOD == "DNN":
    model = dnn.build(
        num_users=NUM_USERS,
        num_movies=NUM_MOVIES,
        latent_dimension=LATENT_DIMENSION,
        is_regularized=IS_REGULARIZED,
        lamda=LAMBDA)

model.summary()

all_ratings = []
for i in range(NUM_MODELS):
    model_file_name = "data/model_{}_{:d}.h5".format(suffix, i)

    model.load_weights(model_file_name)

    all_ratings.append(
        model.predict(
            np.hsplit(x_test, x_test.shape[1]), batch_size=512, verbose=1))
all_ratings = np.array(all_ratings)

if IS_NORMALIZED:
    y_mean_file_name = "data/y_mean.p"
    with open(y_mean_file_name, "rb") as y_mean_file:
        y_mean = pickle.load(y_mean_file)
    user_mean = y_mean["user"]
    movie_mean = y_mean["movie"]
    global_mean = y_mean["global"]

    for i in range(all_ratings.shape[0]):
        for j, x in enumerate(x_test):
            user_id = int(x[0])
            movie_id = int(x[1])
            um = user_mean[user_id]
            mm = movie_mean[movie_id]
            if um != 0 and mm != 0:
                mean = (um + mm + global_mean) / 3
            elif mm != 0:
                mean = mm
            elif um != 0:
                mean = um
            else:
                mean = global_mean
            all_ratings[i][j] += mean
        all_ratings[i][all_ratings[i] < 1] = 1

ratings = np.mean(all_ratings, axis=0)

with open(output_file_name, "w") as output_file:
    output_file.write("TestDataID,Rating\n")
    for i, rating in enumerate(ratings):
        output_file.write("{:d},{:.3f}\n".format(x_test_ids[i], rating[0]))

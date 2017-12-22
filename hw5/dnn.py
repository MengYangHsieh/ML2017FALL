#!/usr/bin/env python3

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate
from keras.layers import Dense, Dropout
from keras.regularizers import l2


def build(num_users,
          num_movies,
          latent_dimension,
          is_regularized=True,
          **kwargs):
    lamda = kwargs["lamda"] if "lamda" in kwargs else 1e-5

    user_id_input = Input(shape=(1, ), name="UserID")
    movie_id_input = Input(shape=(1, ), name="MovieID")

    user_latent = Embedding(
        num_users,
        latent_dimension,
        input_length=1,
        embeddings_initializer="random_uniform",
        embeddings_regularizer=l2(lamda) if is_regularized else None,
        name="UserLatent")(user_id_input)
    movie_latent = Embedding(
        num_movies,
        latent_dimension,
        input_length=1,
        embeddings_initializer="random_uniform",
        embeddings_regularizer=l2(lamda) if is_regularized else None,
        name="MovieLatent")(movie_id_input)

    user_latent = Flatten(name="FlattenedUserLatent")(user_latent)
    movie_latent = Flatten(name="FlattenedMovieLatent")(movie_latent)

    features = Concatenate(
        name="ConcatenatedUserMovieLatent")([user_latent, movie_latent])
    x = Dense(64, activation="elu", name="Dense-1")(features)
    x = Dense(128, activation="elu", name="Dense-2")(x)
    x = Dense(256, activation="elu", name="Dense-3")(x)

    output = Dense(1, activation="elu", name="Rating")(x)

    model = Model([user_id_input, movie_id_input], output)

    model.compile(optimizer="adam", loss="mse", metrics=["mse"])

    return model

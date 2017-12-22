#!/usr/bin/env python3

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dot, Add
from keras.regularizers import l2


def build(num_users=7000,
          num_movies=5000,
          latent_dimension=120,
          is_regularized=True,
          is_biased=True,
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

    output = Dot(1, name="NoBiasedRating")([user_latent, movie_latent])

    model = Model([user_id_input, movie_id_input], output)

    if is_biased:
        bias_input = Input(shape=(1, ), name="GlobalBiasCoef")

        bias = Embedding(
            1,
            1,
            input_length=1,
            embeddings_initializer="random_uniform",
            name="GlobalBias")(bias_input)
        user_bias = Embedding(
            num_users,
            1,
            input_length=1,
            embeddings_initializer="random_uniform",
            embeddings_regularizer=l2(lamda) if is_regularized else None,
            name="UserBias")(user_id_input)
        movie_bias = Embedding(
            num_movies,
            1,
            input_length=1,
            embeddings_initializer="random_uniform",
            embeddings_regularizer=l2(lamda) if is_regularized else None,
            name="MovieBias")(movie_id_input)

        bias = Flatten(name="FlattenedGlobalBias")(bias)
        user_bias = Flatten(name="FlattenedUserBias")(user_bias)
        movie_bias = Flatten(name="FlattenedMovieBias")(movie_bias)

        output = Add(name="Rating")([output, bias, user_bias, movie_bias])

        model = Model([user_id_input, movie_id_input, bias_input], output)

    model.compile(optimizer="adam", loss="mse", metrics=["mse"])

    return model

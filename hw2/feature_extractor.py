#!/usr/bin/env python

import numpy as np
import csv

feature_names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

feature_table = {
    "age": [],
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
        "?",
    ],
    "fnlwgt": [],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ],
    "education-num": [],
    "marital-status": [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
        "?",
    ],
    "relationship": [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ],
    "race": [
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
        "Black",
    ],
    "sex": ["Female", "Male"],
    "capital-gain": [],
    "capital-loss": [],
    "hours-per-week": [],
    "native-country": [
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
        "?",
    ],
}

feature_idx = {
    "age": -1,
    "workclass": -1,
    "fnlwgt": -1,
    "education": -1,
    "education-num": -1,
    "marital-status": -1,
    "occupation": -1,
    "relationship": -1,
    "race": -1,
    "sex": -1,
    "capital-gain": -1,
    "capital-loss": -1,
    "hours-per-week": -1,
    "native-country": -1,
}


def extract_features(raw_training_file_name, raw_testing_file_name,
                     feature_config, is_normalized):
    training_x = []
    training_y = []
    with open(raw_training_file_name, "r") as raw_training_file:
        raw_training_csv = csv.reader(raw_training_file)
        for row in raw_training_csv:
            feature_values = []
            idx = 0
            has_bias = feature_config["bias"]
            if row[0] == "age":
                continue
            if has_bias:
                feature_values.append(1)
                idx += 1
            for i, x in enumerate(row):
                x = x.strip()
                if i < len(feature_names):
                    feature_name = feature_names[i]
                    is_feature_on = len(feature_config[feature_name]) != 0
                    if is_feature_on:
                        if feature_idx[feature_name] == -1:
                            feature_idx[feature_name] = idx
                        subfeatures = feature_table[feature_name]
                        is_discrete = len(subfeatures) != 0
                        if is_discrete:
                            subfeature_values = [
                                0.0 for _ in range(len(subfeatures))
                            ]
                            for j, s in enumerate(subfeatures):
                                if s == x:
                                    subfeature_values[j] = 1.0
                            feature_values += subfeature_values
                            idx += len(subfeatures)
                        else:
                            for p in feature_config[feature_name]:
                                feature_values.append(float(x)**p)
                                idx += 1
                else:
                    training_y.append(1.0 if x == ">50K" else 0.0)
            training_x.append(feature_values)

    training_x = np.matrix(training_x, dtype=np.float64)
    training_y = np.matrix([training_y], dtype=np.float64).transpose()

    testing_x = []
    nth_row = 0
    with open(raw_testing_file_name, "r") as raw_testing_file:
        raw_testing_csv = csv.reader(raw_testing_file)
        for row in raw_testing_csv:
            nth_row += 1
            if nth_row != 1:
                feature_values = []
                has_bias = feature_config["bias"]
                if has_bias:
                    feature_values.append(1)
                for i, x in enumerate(row):
                    x = x.strip()
                    feature_name = feature_names[i]
                    is_feature_on = len(feature_config[feature_name]) != 0
                    if is_feature_on:
                        subfeatures = feature_table[feature_name]
                        is_discrete = len(subfeatures) != 0
                        if is_discrete:
                            subfeature_values = [
                                0.0 for _ in range(len(subfeatures))
                            ]
                            for j, s in enumerate(subfeatures):
                                if s == x:
                                    subfeature_values[j] = 1.0
                            feature_values += subfeature_values
                        else:
                            for p in feature_config[feature_name]:
                                feature_values.append(float(x)**p)
                testing_x.append(feature_values)

    testing_x = np.matrix(testing_x, dtype=np.float64)

    if is_normalized:
        training_x_mean = training_x.mean(axis=0)
        training_x_max_min = training_x.max(axis=0) - training_x.min(axis=0)
        for i, row in enumerate(training_x):
            for k, j in feature_idx.items():
                is_continuous = len(feature_table[k]) == 0
                if is_continuous:
                    for p in feature_config[k]:
                        training_x[i, j + p - 1] = (
                            training_x[i, j + p - 1] -
                            training_x_mean[0, j + p - 1]
                        ) / training_x_max_min[0, j + p - 1]
        for i, row in enumerate(testing_x):
            for k, j in feature_idx.items():
                is_continuous = len(feature_table[k]) == 0
                if is_continuous:
                    for p in feature_config[k]:
                        testing_x[i, j + p - 1] = (
                            testing_x[i, j + p - 1] - training_x_mean[(
                                0,
                                j + p - 1)]) / training_x_max_min[0, j + p - 1]

    return training_x, training_y, testing_x

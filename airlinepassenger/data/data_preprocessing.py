"""
# Created by: marta
# Created on: 04.04.20
"""
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler


def initial_data_preprocessing(data_set):
    data_set.rename(columns={data_set.columns[1]: "passenger_number",
                             data_set.columns[0]: "year_month"}, inplace=True)

    data_set = data_set[:-1]
    data_set["month"] = data_set["year_month"].apply(lambda x: re.sub("\d\d\d\d-0?", "", x))
    return data_set


def create_multivariate_data_set(interim_data_set):
    interim_data_set["previous_passenger_number"] = interim_data_set["passenger_number"].shift(1)
    interim_data_set = interim_data_set[1:]
    final_data_set = interim_data_set.drop(["year_month"], axis=1)
    return final_data_set


def normalize_univariate_series(univariate_data_set):
    univariate_array = univariate_data_set.values
    univariate_array_reshaped = univariate_array.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_univariate_array = scaler.fit_transform(univariate_array_reshaped)
    return scaler, normalized_univariate_array


def split_univariate_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix][0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_univariate_samples(train_set, validation_set, test_set, input_window_size, output_window_size):
    train_samples = split_univariate_sequence(train_set, input_window_size, output_window_size)
    val_samples = split_univariate_sequence(validation_set, input_window_size, output_window_size)
    test_samples = split_univariate_sequence(test_set, input_window_size, output_window_size)

    print(
        f"Number of training samples: {len(train_samples[0])}, validation samples: {len(val_samples[0])}, test samples: {len(test_samples[0])}")
    return train_samples, val_samples, test_samples


def train_val_test_split():
    pass

"""
# Created by: marta
# Created on: 20.04.20
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time


def save_scores(config, scores, parameter_set):
    __data = {"approach": parameter_set[0],
              "input_window_size": parameter_set[1],
              "max_nr_hidden_layer": parameter_set[2],
              "max_nr_lstm_units": parameter_set[3],
              "max_nr_dense_hidden_layer_neurons": parameter_set[4],
              "train_score": scores[0],
              "validation_score": scores[1],
              "test_score": scores[2]
              }

    scores_df = pd.DataFrame(__data, index=[0])

    if os.path.isfile(config.scores):
        scores_df.to_csv(config.scores, mode='a', header=False, index=False)
    else:
        with open(config.scores, 'w') as outfile:
            scores_df.to_csv(outfile, header=True, index=False)


def plot(config, parameter_set, predictions_for_train_denormalized, predictions_for_val_denormalized, predictions_for_test_denormalized):
    interim_data_set = pd.read_csv(config.interim_data)
    nr_samples = interim_data_set["passenger_number"].values.reshape(len(interim_data_set["passenger_number"]), 1)

    # shift train predictions for plotting
    train_predict_plot = np.empty_like(nr_samples)
    train_predict_plot[:, :] = np.nan
    # reshape in order to fill only the training examples
    train_predict_plot[parameter_set[1]:len(predictions_for_train_denormalized) + parameter_set[1],
    :] = predictions_for_train_denormalized

    # shift validation predictions for plotting
    validation_predict_plot = np.empty_like(nr_samples)
    validation_predict_plot[:, :] = np.nan
    validation_predict_plot[
    len(predictions_for_train_denormalized) + (2 * parameter_set[1]):len(predictions_for_train_denormalized) + (
                2 * parameter_set[1]) + len(predictions_for_val_denormalized), :] = predictions_for_val_denormalized

    # shift test predictions for plotting
    test_predict_plot = np.empty_like(nr_samples)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[
    len(predictions_for_train_denormalized) + (2 * parameter_set[1]) + len(predictions_for_val_denormalized):len(
        predictions_for_train_denormalized) + (2 * parameter_set[1]) + len(predictions_for_val_denormalized) + len(
        predictions_for_test_denormalized)] = predictions_for_test_denormalized

    # plot baseline and predictions
    fig = plt.figure(figsize=(18, 10))
    plt.plot(nr_samples)
    plt.plot(train_predict_plot)
    plt.plot(validation_predict_plot)
    plt.plot(test_predict_plot)
    fig_final = plt.gcf()
    plt.show()

    fig_final.savefig(f'{config.plots}plot_{parameter_set[0]}_hp_{parameter_set[1]}_{parameter_set[2]}_{parameter_set[3]}_{parameter_set[4]}_{int((time.time()))}.png', dpi=100)
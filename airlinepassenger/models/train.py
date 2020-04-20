"""
# Created by: marta
# Created on: 04.04.20
"""
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam

# hyperparameter tuning
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping, Callback


def train_model(config, approach, input_window_size, samples, max_nr_dense_hidden_layer,
                max_nr_lstm_units, max_nr_dense_hidden_layer_neurons):
    n_features = samples[0][0].shape[2]

    def build_model(hp):
        model = Sequential()
        # input LSTM layer
        model.add(LSTM(hp.Int("input_units", min_value=5, max_value=max_nr_lstm_units, step=3),
                       activation="relu", input_shape=(input_window_size, n_features), name="lstm"))

        # dense hidden layers with dropouts
        for i in range(hp.Int("n_layers", 0, max_nr_dense_hidden_layer)):
            model.add(Dense(hp.Int(f"dense_{i}_units", min_value=5, max_value=max_nr_dense_hidden_layer_neurons, step=3),
                            activation="relu", name=f"hidden_{i}"))
            for j in range(hp.Int("dropout_layers", 0, 1)):
                model.add(Dropout(hp.Choice(f"drop_rate_{j}", values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                                  name=f"dropout_{i}{j}"))

        # dense output layer
        model.add(Dense(1, name="output"))
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
                      metrics=[metrics.mae])
        return model

    # training & hyperparameter tuning
    LOG_DIR = f"{int((time.time()))}"

    tuner = RandomSearch(
        build_model,
        objective="val_mean_absolute_error",  # evtl. RMSE
        max_trials=5,
        executions_per_trial=3,
        directory=f'{config.models}{approach}_{input_window_size}/tuning_logs',
        project_name=LOG_DIR
    )

    tuner.search(
        x=samples[0][0],
        y=samples[0][1],
        epochs=100,
        batch_size=32,
        validation_data=(samples[1][0], samples[1][1]),
        callbacks=[EarlyStopping('val_loss', patience=5)])

    # get best model
    model = tuner.get_best_models(num_models=1)[0]
    model.save(f"{config.models}{approach}_{input_window_size}")

    return model

"""
# Created by: marta
# Created on: 04.04.20
"""
import airlinepassenger.data.data_preprocessing as mid
import math
from sklearn.metrics import mean_squared_error
import argparse
import logging
import time
from airlinepassenger.config import Configuration
from airlinepassenger.models import train, evaluate

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Airline passengers LSTM models creator")

    parser.add_argument(
        "-c",
        "--config",
        help="config file",
        required=True
    )

    parser.add_argument(
        "-a",
        "--approach",
        choices=["univariate", "multivariate"],
        help="univariate or multivariate",
        required=True
    )

    parser.add_argument(
        "-s",
        "--size",
        choices=[3, 6, 10, 12],
        type=int,
        help="input window size",
        required=True
    )

    parser.add_argument(
        "--max_nr_hidden_layer",
        choices=[0, 1, 2, 3, 4, 5, 6, 7],
        default=2,
        type=int,
        help="Maximal number of dense hidden layers (Recommended=2)"
    )

    parser.add_argument(
        "--max_nr_lstm_units",
        default=50,
        type=int,
        help="Maximal number of LSTM's units (Recommended=50)"
    )

    parser.add_argument(
        "--max_nr_dense_hidden_layer_neurons",
        default=50,
        type=int,
        help="Maximal number of neurons in dense hidden layers (Recommended=50)"
    )

    args = parser.parse_args()

    logger.info("Config file File: %s" % args.config)
    config = Configuration(args.config)

    logger.info("Approach: %s" % args.approach)

    logger.info("Input window size: %s" % args.size)

    logger.info("Maximal number of hidden layers: %s" % args.max_nr_hidden_layer)

    logger.info("Maximal number of LSTM's units: %s" % args.max_nr_lstm_units)

    logger.info("Maximal number of neurons in dense hidden layers: %s" % args.max_nr_dense_hidden_layer_neurons)

    process(config, args.approach, args.size, args.max_nr_hidden_layer, args.max_nr_lstm_units, args.max_nr_dense_hidden_layer_neurons)


def process(config, approach, size, max_nr_hidden_layer, max_nr_lstm_units, max_nr_dense_hidden_layer_neurons):
    logger.info(f"Started creating a {approach} LSTM model.")
    start = time.perf_counter()
    if approach == "univariate":
        # Data preprocessing: Load or create data sets
        train_samples, val_samples, test_samples, scaler = mid.make_univariate_dataset(config, size)

    elif approach == "multivariate":
        train_samples, val_samples, test_samples, scaler = mid.make_multivariate_dataset(config, size)

    samples = [train_samples, val_samples, test_samples]

    # train model
    trained_model = train.train_model(config, approach, size, samples, max_nr_hidden_layer, max_nr_lstm_units, max_nr_dense_hidden_layer_neurons)

    end = time.perf_counter()
    logger.info(f"Model was trained in {end - start:0.4f} seconds.")
    logger.info(f"Make predictions and evaluate.")

    # make predictions
    predictions_for_train = trained_model.predict(train_samples[0])
    predictions_for_validation = trained_model.predict(val_samples[0])
    predictions_for_test = trained_model.predict(test_samples[0])

    # invert predictions
    predictions_for_train_denormalized = scaler.inverse_transform(predictions_for_train)
    true_values_train = scaler.inverse_transform(train_samples[1])

    predictions_for_val_denormalized = scaler.inverse_transform(predictions_for_validation)
    true_values_val = scaler.inverse_transform(val_samples[1])

    predictions_for_test_denormalized = scaler.inverse_transform(predictions_for_test)
    true_values_test = scaler.inverse_transform(test_samples[1])

    # evaluation
    # calculate root mean squared error
    train_score = math.sqrt(mean_squared_error(true_values_train, predictions_for_train_denormalized))
    print('Train Score: %.2f RMSE' % train_score)

    validation_score = math.sqrt(mean_squared_error(true_values_val, predictions_for_val_denormalized))
    print('Validation Score: %.2f RMSE' % validation_score)

    test_score = math.sqrt(mean_squared_error(true_values_test, predictions_for_test_denormalized))
    print('Test Score: %.2f RMSE' % test_score)

    scores = [train_score, validation_score, test_score]
    parameter_set = [approach, size, max_nr_hidden_layer, max_nr_lstm_units, max_nr_dense_hidden_layer_neurons]

    # save scores
    evaluate.save_scores(config, scores, parameter_set)

    # plot predictions
    nr_samples = len(samples[0]) + len(samples[1]) + len(samples[2])
    evaluate.plot(config, parameter_set, predictions_for_train_denormalized, predictions_for_val_denormalized, predictions_for_test_denormalized)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

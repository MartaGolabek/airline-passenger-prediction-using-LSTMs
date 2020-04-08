"""
# Created by: marta
# Created on: 04.04.20
"""
import pandas as pd
import airlinepassenger.data.data_preprocessing as mid
import os.path
import argparse
import logging
from airlinepassenger.config import Configuration

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
        help="input window size",
        required=True
    )

    args = parser.parse_args()

    logger.info("Config file File: %s" % args.config)
    config = Configuration(args.config)

    logger.info("Approach: %s" % args.approach)

    logger.info("Input window size: %s" % args.size)

    process(config, args.approach, args.size)


def process(config, approach, size):
    if approach == "univariate":
        # Data preprocessing: Load or create data sets
        train_samples, val_samples, test_samples = mid.make_univariate_dataset(config, size)
    elif approach == "multivariate":
        train_samples, val_samples, test_samples = mid.make_multivariate_dataset(config, size)


if __name__ == '__main__':
    main()

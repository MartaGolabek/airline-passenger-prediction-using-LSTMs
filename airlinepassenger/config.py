"""
# Created by: marta
# Created on: 08.04.20
"""
import yaml


class Configuration:
    def __init__(self, config_file):
        self.config_path = config_file

        config = yaml.load((open(config_file, "r")))

        self.base_dir = config["base_dir"]

        self.raw_data = self.base_dir + config["raw_data"]
        self.interim_data = self.base_dir + config["interim_data"]
        self.processed_data = self.base_dir + config["processed_data"]
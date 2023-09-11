from src.extractor.utils import read_yaml,create_directories
from src.extractor.constants import CONFIG_FILE_PATH
from src.extractor.entity import DataIngestionConfig
from src.extractor import logging
from box import ConfigBox


class Configuration:
    def __init__(self,config_file_path=CONFIG_FILE_PATH):
        logging.info(f'loading config yaml configuration file: {config_file_path}')
        self.config = read_yaml(config_file_path)
        create_directories([self.config.artifacts_root])
     

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion
        create_directories([config.data_dir])
        create_directories([config.file_dir])
        


        data_ingestion  = DataIngestionConfig(
                             data_dir=config.data_dir,
                             file_dir=config.file_dir)
        return data_ingestion
    
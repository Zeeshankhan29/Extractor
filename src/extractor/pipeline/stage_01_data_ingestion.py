from src.extractor.components import DataIngestion
from src.extractor.config import Configuration
from src.extractor import logger



def main():
    config = Configuration()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.run_streamlit()
    


if __name__ =='__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)

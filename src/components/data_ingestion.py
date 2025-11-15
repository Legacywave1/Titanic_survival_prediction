import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path:  str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
       self.ingestion_config = DataIngestionConfig()
       os.makedirs('artifacts', exist_ok = True)

    def initiate_data_ingestion(self, data_path: str = r'Data\Titanic-Dataset.csv'):
        logging.info('Entered the data ingestion method')
        try:
            df = pd.read_csv(data_path)
            logging.info('Read the dataset as DataFrame')
            df.to_csv(self.ingestion_config.raw_data_path, index = False,header = True)

            return self.ingestion_config.raw_data_path
        except Exception as e:
            raise CustomException(str(e), sys)

if __name__=='__main__':
    get_data = DataIngestion()
    raw_data_path = get_data.initiate_data_ingestion()
    print(f'Data Ingestion Complete and saved at {raw_data_path}')

import joblib
from src.exception import CustomException
from src.logger import logging
import os
import sys


def save_object(file_path, model):
    try:
        logging.info(f'Save {model} pkl using joblib')
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as file:
            joblib.dump(model, file)
            logging.info(f'Save {model} pkl using joblib successful')

    except Exception as e:
        raise CustomException(str(e), sys)


def load_object(file_path):
    try:
        logging.info('Unpacking object')
        with open(file_path, 'rb') as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        raise CustomException(str(e), sys)

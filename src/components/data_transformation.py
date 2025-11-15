import pandas as pd
import numpy as np
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransfomationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    transformed_data_path: str = os.path.join('artifacts', 'data_transformed.csv')

class DataTransformation:
    def __init__(self):
        logging.info('Data Transformation Initialization')
        self.data_transfomation_config = DataTransfomationConfig()

    def get_data_transformed(self):
        try:
            logging.info('Starting get_data_transformation')

            numerical = ['Fare', 'Age', 'FamilySize']
            cat_oh = ['Sex']
            cat_oe = ['Embarked', 'AgeGroup', 'Title', 'FareGroup']
            logging.info('Data types groups to be transformed')
            numerical_pipeline = Pipeline(
                steps = [('num', StandardScaler())
            ])
            cat_oh_pipeline = Pipeline(steps=[
                ('One hot', OneHotEncoder(handle_unknown = 'ignore',drop = 'first', sparse_output = False))
            ])
            cat_oe_pipeline = Pipeline(steps=[
                ('Label encoding', OrdinalEncoder())
            ])

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical),
                    ('cat_oh_pipeline', cat_oh_pipeline, cat_oh),
                    ('cat_oe_pipeline', cat_oe_pipeline, cat_oe)
                ],
                remainder ='drop'
            )

            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer: {str(e)}")
            raise CustomException(str(e), sys)

    def initiate_data_transformation(self, raw_data):
        try:
            df = pd.read_csv(raw_data)
            def title(row):
                return row.split(', ')[1].split('.')[0]
            df['Title'] = df['Name'].apply(title)
            title_mapping = {
                'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
                'Col': 'Rare', 'Major': 'Rare', 'Dr': 'Rare',
                'Rev': 'Rare', 'Jonkheer': 'Rare', 'Don': 'Rare',
                'Sir': 'Rare', 'Lady': 'Rare', 'Countess': 'Rare',
                'Capt': 'Rare', 'Dona': 'Rare'
            }
            df['Title'] = df['Title'].replace(title_mapping)

            age_median = df.groupby(['Title','Pclass'])['Age'].median().reset_index()
            df = df.merge(age_median, on=['Title', 'Pclass'], suffixes = ('', '_median'))

            #df1.columns
            df['Age'] = df['Age'].fillna(df['Age_median'])
            df = df.drop('Age_median', axis = 1)

            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            def Groupage(x):
                if x < 12:
                    return 'Child'
                elif x <= 45:
                    return 'Adult'
                else:
                    return 'Senior'
            df['AgeGroup'] = df['Age'].apply(Groupage)
            df['FarePerPerson'] = df['Fare'] / df['FamilySize'].replace(0,1)
            df['FareGroup'] = pd.qcut(df['FarePerPerson'], 4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
            df = df.drop(columns=['PassengerId', 'Name', 'Ticket'], axis = 1)

            preprocessor = self.get_data_transformed()
            target_column_name = 'Survived'

            input_feature_train_df = df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = df[target_column_name]

            logging.info(f"Training features shape: {input_feature_train_df.shape}")

            logging.info('Applying preprocessing object on training data')
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

            feature_name = preprocessor.get_feature_names_out()
            self.save_transformed_data(input_feature_train_arr, target_feature_train_df, feature_name, self.data_transfomation_config.transformed_data_path)

            save_object(self.data_transfomation_config.preprocessor_obj_file_path, preprocessor)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            return (
                train_arr,
                self.data_transfomation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(str(e), sys)

    def save_transformed_data(self, X, y, columns, path):
        try:
            logging.info(f"Saving transformed data to: {path}")
            logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            transformed_df = pd.DataFrame(X, columns = columns)
            transformed_df['Survival'] = y.values
            os.makedirs(os.path.dirname(path), exist_ok = True)
            transformed_df.to_csv(path, index = False)

        except Exception as e:
            raise CustomException(str(e), sys)

if __name__ == '__main__':
    try:
        data = 'artifacts/raw_data.csv'
        logging.info("Starting data transformation process")
        data_transformation = DataTransformation()
        train_arr, preprocessor_path = data_transformation.initiate_data_transformation(data)

        print(f'Training data shape: {train_arr.shape}')
        print(f'Preprocessor saved at: {preprocessor_path}')
    except Exception as e:
        raise CustomException(str(e), sys)

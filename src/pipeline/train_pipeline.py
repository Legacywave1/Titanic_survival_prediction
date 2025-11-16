import os
import sys
import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings('ignore')


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

@dataclass
class DataTransfomationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    transformed_train_data_path: str = os.path.join('artifacts', 'train_data_transformed.csv')
    transformed_test_data_path: str = os.path.join('artifacts', 'test_data_transformed.csv')

class DataTransformation:
    def __init__(self):
        logging.info('Data Transformation Initialization')
        self.data_transfomation_config = DataTransfomationConfig()



    def get_data_transformed(self):
        try:
            logging.info('Starting get_data_transformation')

            numerical = ['Fare', 'Age', 'FamilySize', 'FarePerPerson']
            cat_oh = ['Sex', 'IsAlone']
            cat_oe = ['Embarked', 'AgeGroup', 'Title', 'FareGroup', 'Pclass']

            logging.info('Data types groups to be transformed')
            numerical_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('num', StandardScaler())
            ])
            cat_oh_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('One hot', OneHotEncoder(handle_unknown = 'ignore',drop = 'first', sparse_output = False))
            ])
            cat_oe_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('Ordinal encoding', OrdinalEncoder())

            ])

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical),
                    ('cat_oh_pipeline', cat_oh_pipeline, cat_oh),
                    ('cat_oe_pipeline', cat_oe_pipeline, cat_oe),
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
            titles = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            df['Title'] = titles
            title_mapping = {
                'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
                'Col': 'Rare', 'Major': 'Rare', 'Dr': 'Rare',
                'Rev': 'Rare', 'Jonkheer': 'Rare', 'Don': 'Rare',
                'Sir': 'Rare', 'Lady': 'Rare', 'Countess': 'Rare',
                'Capt': 'Rare', 'Dona': 'Rare'
            }
            df['Title'] = df['Title'].apply(lambda x: title_mapping.get(x,x)if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)
            df['Title'] = df['Title'].replace(title_mapping)
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            df['FarePerPerson'] = df['Fare'] / df['FamilySize'].replace(0,1)
            df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch'], axis = 1)
            train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42, stratify= df['Survived'])

            age_median = train_df.groupby(['Title','Pclass'])['Age'].median()
            train_df['Age'] = train_df.apply(lambda row: age_median[row['Title'],row['Pclass']] if pd.isna(row['Age']) else row['Age'], axis = 1)
            test_df['Age'] = test_df.apply(lambda row: age_median[row['Title'],row['Pclass']] if pd.isna(row['Age']) else row['Age'], axis = 1)


            def Groupage(x):
                if x < 12:
                    return 'Child'
                elif x <= 45:
                    return 'Adult'
                else:
                    return 'Senior'
            train_df['AgeGroup'] = train_df['Age'].apply(Groupage)
            test_df['AgeGroup'] = test_df['Age'].apply(Groupage)
            fare_bin = pd.qcut(train_df['FarePerPerson'], 4, labels=False, retbins = True, duplicates = 'drop')[1]
            labels = ['Low', 'Medium', 'High', 'VeryHigh']
            train_df['FareGroup'] = pd.cut(x = train_df['FarePerPerson'], bins=fare_bin, labels= labels, include_lowest = True)

            test_df['FareGroup'] = pd.cut(x = test_df['FarePerPerson'], bins=fare_bin, labels = labels, include_lowest = True)

            preprocessor = self.get_data_transformed()
            target_column_name = 'Survived'

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Training features shape: {input_feature_train_df.shape}")

            logging.info('Applying preprocessing object on training data')
            preprocessor.fit(input_feature_train_df)
            input_feature_train_arr = preprocessor.transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            feature_name = preprocessor.get_feature_names_out()
            self.save_transformed_data(input_feature_train_arr, target_feature_train_df, feature_name, self.data_transfomation_config.transformed_train_data_path)
            self.save_transformed_data(input_feature_test_arr, target_feature_test_df, feature_name, self.data_transfomation_config.transformed_test_data_path)

            save_object(self.data_transfomation_config.preprocessor_obj_file_path, preprocessor)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            return (
                self.data_transfomation_config.transformed_train_data_path,
                self.data_transfomation_config.transformed_test_data_path,
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


@dataclass
class ModelTrainingConfig:
    train_model_path: str = os.path.join('artifacts', 'trained_model.pkl')
    optuna_study_path: str = os.path.join('artifacts', 'optuna_study.pkl')
    mlflow_path: str = 'mlrun'

class ModelTraining:
    def __init__(self):
        self.config = ModelTrainingConfig()

        mlflow.set_tracking_uri(self.config.mlflow_path)
        mlflow.set_experiment('Titanic Survival Predictor')

        logging.info(f'MlFlow tracking URI: {self.config.mlflow_path}')

    def objective(self, trial, X_train, y_train, model_name):
        try:
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            if not isinstance(y_train, pd.Series):
                y_train = pd.Series(y_train)
            if model_name == 'Random Forest':
                n_estimators = trial.suggest_int('rf_n_estimators', 50, 200)
                max_depth = trial.suggest_int('rf_max_depth', 3, 20)
                min_samples_split = trial.suggest_int('rf_min_samples_split',2,20)
                min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 10)
                model = RandomForestClassifier(
                    n_estimators = n_estimators,
                    max_depth = max_depth,
                    min_samples_leaf= min_samples_leaf,
                    min_samples_split = min_samples_split,
                    random_state = 42
                )
            elif model_name == 'XGB':
                booster = trial.suggest_categorical('XGB_booster', ['gbtree', 'gblinear', 'dart'])
                n_estimators = trial.suggest_int('XGB_n_estimators', 50, 200)
                learning_rate = trial.suggest_float('XGB_learning_rate', 0.01, 0.5)
                max_depth = trial.suggest_int("XGB_max_depth", 3, 10)
                model = XGBClassifier(
                    booster = booster,
                    n_estimators = n_estimators,
                    learning_rate = learning_rate,
                    max_depth = max_depth,
                    random_state = 42
                )
            else:
                model = LogisticRegression()


            skf = StratifiedKFold(n_splits = 7, shuffle = True, random_state = 1)
            cv_scores = []
            feature_importances = []
            all_true_vals = []
            all_pred_vals = []
            test_prediction = np.zeros(len(X_train))


            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train,y_train)):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_train_fold, y_train_fold)
                val_preds = model.predict(X_val_fold)
                all_true_vals.extend(y_val_fold)
                all_pred_vals.extend(val_preds)

                test_prediction[val_idx] = val_preds

                fold_accuracy = accuracy_score(y_val_fold, val_preds)
                cv_scores.append(fold_accuracy)

                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
                elif hasattr(model, 'coef_'):
                    feature_importances.append(model.coef_[0])
            print('Overall Classification Report')
            fold_report = classification_report(all_true_vals, all_pred_vals, target_names = ['Died', 'Survived'])
            print(fold_report)

            print('Overall Confusion Matrix Report')
            fold_report = confusion_matrix(all_true_vals, all_pred_vals)
            print(fold_report)

            return np.mean(cv_scores)
        except Exception as e:
            return -1000

    def optimize_models_with_optuna(self, X_train, y_train,models, n_trial = 20):
        optimized_models = {}
        best_scores = {}
        for model_name in models.keys():
            with mlflow.start_run(run_name = f'{model_name}_Optimization', nested = True):
                if model_name == 'Logistic Regression':
                    skf = StratifiedKFold(n_splits = 7, shuffle = True, random_state = 1)
                    cross_val = []
                    test_predictions = np.zeros(len(X_train))
                    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        lr = LogisticRegression()
                        lr.fit(X_train_fold, y_train_fold)
                        pred = lr.predict(X_val_fold)

                        test_predictions[val_idx] = pred
                        fold_accuracy = accuracy_score(y_val_fold, pred)
                        cross_val.append(fold_accuracy)

                    optimized_models[model_name] = LogisticRegression()
                    score = np.mean(cross_val)
                    best_scores[model_name] = score
                    mlflow.log_metric("Best_cv_score", score)
                    mlflow.log_param('Model_type', 'Logistic Regression')

                    continue

                study = optuna.create_study(direction = 'maximize')
                def create_objective(name):
                    return lambda trial: self.objective(trial, X_train, y_train, name)
                study.optimize(create_objective(model_name), n_trials = n_trial)
                if study.best_value == -1000.0:
                    logging.warning(f'All trials failed for {model_name}, using default parameters')
                    optimized_models[model_name] = models[model_name]
                    best_scores[model_name] = -1000.0
                    continue
                best_params = study.best_params
                best_score = study.best_value
                mlflow.log_params(best_params)
                mlflow.log_metric('Best_cv_score', best_score)
                mlflow.log_param('Model_type', model_name)
                mlflow.log_param('n_trial', n_trial)
                if model_name == 'Random Forest':
                    optimized_model =RandomForestClassifier(
                        n_estimators = best_params['rf_n_estimators'],
                        max_depth = best_params['rf_max_depth'],
                        min_samples_leaf= best_params['rf_min_samples_leaf'],
                        min_samples_split = best_params['rf_min_samples_split'],
                        random_state = 42
                    )
                elif model_name == 'XGB':
                    optimized_model = XGBClassifier(
                        booster = best_params['XGB_booster'],
                        n_estimators = best_params['XGB_n_estimators'],
                        learning_rate = best_params['XGB_learning_rate'],
                        max_depth = best_params['XGB_max_depth'],
                        random_state = 42
                    )
                optimized_models[model_name] = optimized_model
                best_scores[model_name] = best_score
                mlflow.sklearn.log_model(optimized_model, f"{model_name.lower().replace(' ', '_')}_model")
        return optimized_models, best_scores

    def initiate_train(self, train_path, test_path, use_optuna = True, n_trials = 20):
        try:
            logging.info('The initiating about to commence')
            with mlflow.start_run(run_name = 'Model_Training_Pipeline') as run:
                self.current_run_id = run.info.run_id
                mlflow.log_param('train_path', train_path)
                mlflow.log_param('use_optuna', use_optuna)
                mlflow.log_param('n_trials', n_trials)

                train = pd.read_csv(train_path)

                X_train = train.drop('Survival', axis = 1)
                y_train = train['Survival']

                X_train = pd.DataFrame(X_train, columns=train.drop('Survival', axis=1).columns)

                base_models = {
                    'Random Forest': RandomForestClassifier(random_state = 42),
                    'XGB': XGBClassifier(random_state = 42),
                    'Logistic Regression': LogisticRegression(random_state = 42)
                }

                if use_optuna:
                    models, scores = self.optimize_models_with_optuna(X_train, y_train, base_models, n_trials)
                else:
                    models = base_models

                    scores = {}
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        cv_scores = cross_val_score(model, X_train, y_train, cv = 5)
                        scores[name] = np.mean(cv_scores)

                best_model_name = max(scores, key = scores.get)
                best_model = models[best_model_name]
                best_cv_score = scores[best_model_name]
                logging.info(f"Best model: {best_model_name} with score: {best_cv_score}")

                test = pd.read_csv(test_path)
                X_test = test.drop('Survival', axis = 1)
                y_test = test['Survival']
                best_model.fit(X_train, y_train)
                test_preds = best_model.predict(X_test)
                test_set_accuracy = accuracy_score(y_test, test_preds)
                classification = classification_report(y_test,test_preds, target_names = ['Died', 'Survived'])
                print(classification)
                mlflow.log_metric('test_set_accuracy', test_set_accuracy)
                mlflow.log_text(classification, 'classification_report.txt')
                mlflow.log_metric('best_cv_score', best_cv_score)
                mlflow.log_param('best_model', best_model_name)

                save_object(self.config.train_model_path, best_model)
                mlflow.log_artifact(self.config.train_model_path)


                self.registered_model_name = f'Titanic_predictor_{best_model_name}'

                mlflow.sklearn.log_model(
                    best_model,
                    'Best_model',
                    registered_model_name = f'Titanic_predictor_{best_model_name}'
                )

                return best_model_name, best_cv_score, self.current_run_id, self.registered_model_name

        except Exception as e:
            raise CustomException(str(e), sys)


@dataclass
class ModelPromotionConfig:
    staging_alias: str = 'Staging'
    production_alias: str = 'Production'
    champion_threshold: float = 0.7
    challenger_threshold: float = 0.02

class ModelPromotion:
    def __init__(self):
        self.promotion_config = ModelPromotionConfig()
        self.client = MlflowClient()

    def get_current_production_model(self, model_name):
        try:
            production_models = self.client.get_model_version_by_alias(model_name, self.promotion_config.production_alias)
            return production_models

        except Exception:
            return 'No current Model in production'

    def evaluate_model_for_promotion(self, new_model_score, current_model_info = None):
        try:
            if new_model_score < self.promotion_config.champion_threshold:
                logging.info(f'Model score {new_model_score:.4f} below promotion threshold')
                return False, 'Below quality threshold'

            if current_model_info is None:
                return True, 'First production model'

            current_model_run = self.client.get_run(current_model_info.run_id)
            current_score = current_model_run.data.metrics.get('Best_cv_score', 0)

            improvement = new_model_score - current_score
            if improvement >= self.promotion_config.challenger_threshold:
                return True, f'Significant improvement: {improvement:.4f}'
            else:
                return False, f'Insufficient improvement: {improvement:.4f}'

        except Exception as e:
            return False, f'Evaluation error: {str(e)}'


    def promote_model(self, model_name, version, promotion_type = 'staging'):
        try:
            alias = self.promotion_config.staging_alias if promotion_type == 'staging' else self.promotion_config.production_alias
            self.client.set_model_version_tag(
                model_name,
                version,
                'promotion_status',
                promotion_type
            )
            self.client.set_registered_model_alias(
                model_name,
                alias,
                str(version)
            )
            return True
        except Exception as e:
            logging.error(f'Error promoting model: {e}')
            return False

    def find_model_version_by_run_id(self, model_name, run_id, max_retries = 5, delay = 2):
        for attempt in range(max_retries):
            try:
                model_versions = self.client.search_model_versions(f"run_id='{run_id}")
                if model_versions:
                    return model_versions[0]
                logging.info(f'Model version not found yet (attempt {attempt + 1}/{max_retries}, waiting {delay} seconds...')
                time.sleep(delay)
            except Exception as e:
                logging.warning(f"Error searching for model version (attempt {attempt + 1}): {e}")
                time.sleep(delay)

            try:
                    model_versions = self.client.search_model_versions(f"name='{model_name}'")
                    if model_versions:
                        latest_version = max(model_versions, key = lambda x: x.version)
                        logging.info(f"Using latest version {latest_version.version} for model {model_name}")
                        return latest_version
            except Exception as e:
                logging.error(f"Error getting latest model version: {str(e)}")
        return None

    def initiate_model_promotion(self, best_model_name, best_model_score, run_id, registered_model_name):
        try:
            current_production = self.get_current_production_model(registered_model_name)
            should_promote, reason = self.evaluate_model_for_promotion(
                best_model_score,
                current_production
            )
            model_version = self.find_model_version_by_run_id(registered_model_name, run_id)
            if not model_version:
                return False, 'No model version found'
            model_name = model_version.name
            version = model_version.version
            promotion_result = {
                'model_name': model_name,
                'version': version,
                'score': best_model_score,
                'Promoted_to_staging': False,
                'Promoted_to_production': False,
                'reason': reason
            }

            if self.promote_model(model_name, version, 'staging'):
                promotion_result['Promoted_to_staging'] = True

            if should_promote:
                if self.promote_model(model_name, version, 'production'):
                    promotion_result['Promoted_to_production'] = True

                    if current_production:
                        self.client.set_model_version_tag(
                            model_name,
                            current_production.version,
                            'archived',
                            'true'
                        )
            return True, promotion_result

        except Exception as e:
            logging.error(f'Model promotion failed {str(e)}')

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTraining()
        self.model_promoter = ModelPromotion()
    def run_pipeline(self, use_optuna = True, n_trials = 25, enable_promotion = True):
        try:
            logging.info('Starting complete training pipeline')
            logging.info('Step 1: Data Ingestion')
            data_path = self.data_ingestion.initiate_data_ingestion()

            logging.info('Step 2: Data Transformation')
            train_path, test_path, preprocessor_path = self.data_transformation.initiate_data_transformation(data_path)

            logging.info('Step 3: Model Training')
            best_model_name, best_score, run_id, reg_name = self.model_trainer.initiate_train(train_path,test_path, use_optuna, n_trials)

            promotion_results = None
            if enable_promotion and run_id and reg_name:
                logging.info('Step 4: Model Promotion')
                promotion_success, promotion_results = self.model_promoter.initiate_model_promotion(
                    best_model_name, best_score, run_id, reg_name
                )
                if not promotion_success:
                    logging.warning(f'Model promotion failed: {promotion_results}')
            logging.info("Training pipeline Completed Sucessfully!")

            return {
                'best_model_name': best_model_name,
                'best_score' : best_score,
                'preprocessor_path': preprocessor_path,
                'promotion_results':promotion_results
            }

        except Exception as e:
            logging.error(f'Error in training pipeline: {e}')
            raise CustomException(str(e), sys)

if __name__ == '__main__':
    try:
        pipeline = TrainingPipeline()
        results = pipeline.run_pipeline(use_optuna= True, n_trials = 25, enable_promotion = True)
        print(f"Best Model: {results['best_model_name']}")
        print(f"Best Model Score: {results['best_score']}")
        if results['promotion_results']:
            print(f"Promotion Status: {results['promotion_results']}")
        print('Congratulations!')
    except Exception as e:
        raise CustomException(str(e), sys)

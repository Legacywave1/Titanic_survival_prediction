from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import optuna
import mlflow
import mlflow.sklearn
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import os
import sys


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

                feature_importances.append(model.feature_importances_)
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

    def initiate_train(self, train_path, use_optuna = True, n_trials = 20):
        try:
            logging.info('The initiating about to commence')
            with mlflow.start_run(run_name = 'Model_Training_Pipeline'):
                mlflow.log_param('train_path', train_path)
                mlflow.log_param('use_optuna', use_optuna)
                mlflow.log_param('n_trials', n_trials)

                train = pd.read_csv(train_path)

                            # Check for NaN values
                if train.isnull().any().any():
                    logging.warning("NaN values found in training data!")
                    # Option 1: Fill NaN values
                    train = train.fillna(train.median(numeric_only=True))  # For numerical
                    # For categorical, you might need mode or specific handling

                    # Option 2: Drop rows with NaN (if few)
                    # train = train.dropna()

                    logging.info(f"After handling NaN - Data shape: {train.shape}")
                X_train = train.drop('Survival', axis = 1)
                y_train = train['Survival']

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
                best_score = scores[best_model_name]
                logging.info(f"Best model: {best_model_name} with score: {best_score}")

                save_object(self.config.train_model_path, best_model)
                mlflow.log_artifact(self.config.train_model_path)
                mlflow.log_metric('best_model_score', best_score)
                mlflow.log_param('best_model', best_model_name)
                return self.config.train_model_path, best_model_name, best_score




        except Exception as e:
            raise CustomException(str(e), sys)
if __name__ == '__main__':
    model_trainer = ModelTraining()
    model_path, model_name, score = model_trainer.initiate_train(r'artifacts\data_transformed.csv')
    print(f'Training Complete, man. Best model: {model_name}, with score: {score}. I think you have done enough and should go get some sleep man')

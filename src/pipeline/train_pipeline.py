import os
import sys
import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
import time
import shutil
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --- DYNAMIC PATH SETUP (Matches Predict Pipeline) ---
# 1. Get the absolute path of this script file
current_file_path = Path(__file__).resolve()

# 2. Go up 3 levels: src/pipeline/train_pipeline.py -> src/pipeline -> src -> ROOT
PROJECT_ROOT = current_file_path.parent.parent.parent

# 3. Define where mlrun is relative to ROOT
MLRUN_PATH = PROJECT_ROOT / "mlrun"

# 4. Generate the URI (handles file:///C:/... or file:///app/...) automatically
TRACKING_URI = MLRUN_PATH.as_uri()

# 5. Set the URI globally
mlflow.set_tracking_uri(TRACKING_URI)

# Ensure src can be imported
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        os.makedirs('artifacts', exist_ok=True)

    def initiate_data_ingestion(self, data_path: str = r'Data/Titanic-Dataset.csv'):
        logging.info('Data ingestion started')
        try:
            # Fix: Handle data path relative to project root if needed
            if not os.path.isabs(data_path):
                 data_path = os.path.join(PROJECT_ROOT, data_path)

            if not os.path.exists(data_path):
                logging.warning(f'Data file not found at {data_path}. Creating dummy dataset.')
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                dummy_df = pd.DataFrame({
                    'PassengerId': [1, 2, 3], 'Survived': [0, 1, 1], 'Pclass': [3, 1, 3],
                    'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'Heikkinen, Miss. Laina'],
                    'Sex': ['male', 'female', 'female'], 'Age': [22, 38, 26], 'SibSp': [1, 1, 0],
                    'Parch': [0, 0, 0], 'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
                    'Fare': [7.25, 71.2833, 7.925], 'Cabin': [np.nan, 'C85', np.nan], 'Embarked': ['S', 'C', 'S']
                })
                dummy_df.to_csv(data_path, index=False)

            df = pd.read_csv(data_path)
            logging.info('Dataset loaded into DataFrame')
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            return self.ingestion_config.raw_data_path
        except Exception as e:
            raise CustomException(str(e), sys)

@dataclass
class DataTransfomationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    transformed_train_data_path: str = os.path.join('artifacts', 'train_data_transformed.csv')
    transformed_test_data_path: str = os.path.join('artifacts', 'test_data_transformed.csv')
    age_map_path: str = os.path.join('artifacts', 'age_median.pkl')
    fare_bins_path: str = os.path.join('artifacts', 'fare_bins.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransfomationConfig()

    def get_preprocessor(self):
        logging.info('Building preprocessor')
        numerical = ['Fare', 'Age', 'FamilySize', 'FarePerPerson']
        cat_oh = ['Sex', 'IsAlone']
        cat_oe = ['Embarked', 'AgeGroup', 'Title', 'FareGroup', 'Pclass']
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_oh_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])
        cat_oe_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder())
        ])
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical),
            ('cat_oh', cat_oh_pipeline, cat_oh),
            ('cat_oe', cat_oe_pipeline, cat_oe)
        ], remainder='drop')
        return preprocessor

    def initiate_data_transformation(self, raw_path):
        logging.info('Data transformation started')
        try:
            df = pd.read_csv(raw_path)
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            title_mapping = {
                'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Col': 'Rare', 'Major': 'Rare',
                'Dr': 'Rare', 'Rev': 'Rare', 'Jonkheer': 'Rare', 'Don': 'Rare', 'Sir': 'Rare',
                'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Dona': 'Rare'
            }
            df['Title'] = df['Title'].apply(lambda x: title_mapping.get(x, x) if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)
            df['Title'] = df['Title'].replace(title_mapping)
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            df['FarePerPerson'] = df['Fare'] / df['FamilySize'].replace(0, 1)
            df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Cabin'], errors='ignore')
            if 'Survived' not in df.columns:
                raise CustomException("Missing 'Survived' column", sys)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Survived'])
            age_median = train_df.groupby(['Title', 'Pclass'])['Age'].median()
            save_object(self.config.age_map_path, age_median)
            train_df['Age'] = train_df.apply(lambda r: age_median.get((r['Title'], r['Pclass']), train_df['Age'].median()) if pd.isna(r['Age']) else r['Age'], axis=1)
            test_df['Age'] = test_df.apply(lambda r: age_median.get((r['Title'], r['Pclass']), train_df['Age'].median()) if pd.isna(r['Age']) else r['Age'], axis=1)
            def group_age(x):
                if x < 12: return 'Child'
                elif x <= 45: return 'Adult'
                else: return 'Senior'
            train_df['AgeGroup'] = train_df['Age'].apply(group_age)
            test_df['AgeGroup'] = test_df['Age'].apply(group_age)
            unique_fares = train_df['FarePerPerson'].nunique()
            n_bins = min(4, max(2, unique_fares - 1)) if unique_fares > 1 else 2
            fare_bins = pd.qcut(train_df['FarePerPerson'], n_bins, duplicates='drop', retbins=True)[1] if unique_fares > 1 else pd.cut(train_df['FarePerPerson'], 2, retbins=True)[1]
            save_object(self.config.fare_bins_path, fare_bins)
            labels = ['Low', 'Medium', 'High', 'VeryHigh'][:len(fare_bins)-1] or ['Default']
            train_df['FareGroup'] = pd.cut(train_df['FarePerPerson'], bins=fare_bins, labels=labels, include_lowest=True)
            test_df['FareGroup'] = pd.cut(test_df['FarePerPerson'], bins=fare_bins, labels=labels, include_lowest=True)
            preprocessor = self.get_preprocessor()
            X_train = train_df.drop('Survived', axis=1)
            y_train = train_df['Survived']
            X_test = test_df.drop('Survived', axis=1)
            y_test = test_df['Survived']
            preprocessor.fit(X_train)
            train_arr = preprocessor.transform(X_train)
            test_arr = preprocessor.transform(X_test)
            cols = preprocessor.get_feature_names_out()
            self._save_array(train_arr, y_train, cols, self.config.transformed_train_data_path)
            self._save_array(test_arr, y_test, cols, self.config.transformed_test_data_path)
            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            return self.config.transformed_train_data_path, self.config.transformed_test_data_path, self.config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(str(e), sys)

    def _save_array(self, X, y, cols, path):
        df = pd.DataFrame(X, columns=cols)
        df['Survival'] = y.values
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

@dataclass
class ModelTrainingConfig:
    model_path: str = os.path.join('artifacts', 'trained_model.pkl')
    mlflow_dir: str = 'mlrun'

class ModelTraining:
    def __init__(self):
        self.config = ModelTrainingConfig()
        # Ensure we create the directory pointed to by our dynamic path
        MLRUN_PATH.mkdir(parents=True, exist_ok=True)

        # Use the global URI we set at the top
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment('Titanic Survival Predictor')

    def objective(self, trial, X, y, name):
        try:
            if name == 'Random Forest':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('rf_n_estimators', 50, 200),
                    max_depth=trial.suggest_int('rf_max_depth', 3, 20),
                    min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 10),
                    random_state=42
                )
            elif name == 'XGB':
                model = XGBClassifier(
                    booster=trial.suggest_categorical('xgb_booster', ['gbtree', 'gblinear', 'dart']),
                    n_estimators=trial.suggest_int('xgb_n_estimators', 50, 200),
                    learning_rate=trial.suggest_float('xgb_learning_rate', 0.01, 0.5),
                    max_depth=trial.suggest_int('xgb_max_depth', 3, 10),
                    random_state=42,
                    eval_metric='logloss'
                )
            else:
                model = LogisticRegression(random_state=42)
            skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=1)
            scores = []
            for _, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
                model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                pred = model.predict(X.iloc[val_idx])
                scores.append(accuracy_score(y.iloc[val_idx], pred))
            return np.mean(scores)
        except Exception as e:
            logging.error(f'Optuna trial failed: {e}')
            return -1000

    def optimize(self, X, y, models, n_trials):
        optimized, scores = {}, {}
        for name in models:
            with mlflow.start_run(run_name=f'{name}_opt', nested=True):
                if name == 'Logistic Regression':
                    lr = LogisticRegression(random_state=42)
                    cv = cross_val_score(lr, X, y, cv=7, scoring='accuracy')
                    optimized[name] = lr
                    scores[name] = cv.mean()
                    mlflow.log_metric('cv_score', scores[name])
                    continue
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda t: self.objective(t, X, y, name), n_trials=n_trials)
                if study.best_value == -1000:
                    optimized[name] = models[name]
                    scores[name] = -1000
                    continue
                params = study.best_params
                mlflow.log_params(params)
                mlflow.log_metric('cv_score', study.best_value)
                if name == 'Random Forest':
                    optimized[name] = RandomForestClassifier(**{k[3:]: v for k, v in params.items()}, random_state=42)
                elif name == 'XGB':
                    optimized[name] = XGBClassifier(**{k[4:]: v for k, v in params.items()}, random_state=42, eval_metric='logloss')
                scores[name] = study.best_value
        return optimized, scores

    def initiate_train(self, train_path, test_path, use_optuna=True, n_trials=20):
        logging.info('Model training started')
        with mlflow.start_run(run_name='Training_Run') as run:
            self.run_id = run.info.run_id
            train = pd.read_csv(train_path)
            X_train = train.drop('Survival', axis=1)
            y_train = train['Survival']
            base = {
                'Random Forest': RandomForestClassifier(random_state=42),
                'XGB': XGBClassifier(random_state=42, eval_metric='logloss'),
                'Logistic Regression': LogisticRegression(random_state=42)
            }
            if use_optuna:
                models, scores = self.optimize(X_train, y_train, base, n_trials)
            else:
                models, scores = base, {}
                for n, m in base.items():
                    cv = cross_val_score(m, X_train, y_train, cv=5)
                    scores[n] = cv.mean()
            best_name = max(scores, key=scores.get)
            best_model = models[best_name]
            best_score = scores[best_name]
            test = pd.read_csv(test_path)
            X_test = test.drop('Survival', axis=1)
            y_test = test['Survival']
            best_model.fit(X_train, y_train)
            pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            report = classification_report(y_test, pred, target_names=['Died', 'Survived'])
            mlflow.log_metric('test_accuracy', acc)
            mlflow.log_text(report, 'report.txt')
            mlflow.log_metric('cv_score', best_score)
            mlflow.log_param('best_model', best_name)
            save_object(self.config.model_path, best_model)
            os.makedirs('model_artifacts', exist_ok=True)
            for f in ['preprocessor.pkl', 'age_median.pkl', 'fare_bins.pkl']:
                shutil.copy(os.path.join('artifacts', f), f'model_artifacts/{f}')
            shutil.copy(self.config.model_path, 'model_artifacts/trained_model.pkl')
            mlflow.log_artifacts('model_artifacts', artifact_path='model_artifacts')
            shutil.rmtree('model_artifacts')
            reg_name = f'Titanic_predictor_{best_name.replace(" ", "_")}'
            mlflow.sklearn.log_model(best_model, 'model', registered_model_name=reg_name, input_example=X_train.head(1))
            return best_name, best_score, self.run_id, reg_name

@dataclass
class ModelPromotionConfig:
    staging: str = 'Staging'
    production: str = 'Production'
    min_score: float = 0.7
    min_improvement: float = 0.02

class ModelPromotion:
    def __init__(self):
        self.cfg = ModelPromotionConfig()
        # Pass the global URI explicitly to be safe
        self.client = MlflowClient(tracking_uri=TRACKING_URI)

    def get_prod_version(self, name):
        try:
            return self.client.get_model_version_by_alias(name, self.cfg.production)
        except:
            return None

    def evaluate(self, new_score, prod_info):
        if new_score < self.cfg.min_score:
            return False, 'below threshold'
        if not prod_info:
            return True, 'first model'
        prod_run = self.client.get_run(prod_info.run_id)
        old_score = prod_run.data.metrics.get('cv_score', 0)
        if new_score - old_score >= self.cfg.min_improvement:
            return True, f'improved by {new_score - old_score:.4f}'
        return False, 'insufficient improvement'

    def promote(self, name, version, stage):
        alias = self.cfg.staging if stage == 'staging' else self.cfg.production
        self.client.set_registered_model_alias(name, alias, str(version))
        logging.info(f'Promoted {name} v{version} to {alias}')
        return True

    def find_version(self, run_id, retries=10, delay=5):
        for _ in range(retries):
            vers = self.client.search_model_versions(f"run_id='{run_id}'")
            if vers:
                return vers[0]
            time.sleep(delay)
        return None

    def initiate(self, best_name, best_score, run_id, reg_name):
        prod = self.get_prod_version(reg_name)
        promote, reason = self.evaluate(best_score, prod)
        version_info = self.find_version(run_id)
        if not version_info:
            return False, 'version not found'
        v = version_info.version
        self.promote(reg_name, v, 'staging')
        if promote:
            if prod:
                self.client.delete_registered_model_alias(reg_name, self.cfg.production)
            self.promote(reg_name, v, 'production')
        return True, {'model': reg_name, 'version': v, 'score': best_score, 'to_staging': True, 'to_production': promote, 'reason': reason}

class TrainingPipeline:
    def __init__(self):
        self.ingest = DataIngestion()
        self.transform = DataTransformation()
        self.train = ModelTraining()
        self.promote = ModelPromotion()

    def run(self, use_optuna=True, n_trials=25, promote=True):
        logging.info('Training pipeline started')
        raw = self.ingest.initiate_data_ingestion()
        train_path, test_path, _ = self.transform.initiate_data_transformation(raw)
        best_name, best_score, run_id, reg_name = self.train.initiate_train(train_path, test_path, use_optuna, n_trials)
        promo_res = None
        if promote and run_id and reg_name:
            success, promo_res = self.promote.initiate(best_name, best_score, run_id, reg_name)
            if not success:
                logging.warning(f'Promotion failed: {promo_res}')
        logging.info('Training pipeline completed')
        return {
            'best_model': best_name,
            'score': best_score,
            'registered_name': reg_name,
            'promotion': promo_res
        }

if __name__ == '__main__':
    pipeline = TrainingPipeline()
    res = pipeline.run(use_optuna=True, n_trials=5, promote=True)
    print(f"Best Model: {res['best_model']}")
    print(f"Score: {res['score']:.4f}")
    if res['promotion']:
        print(f"Promotion: {res['promotion']}")

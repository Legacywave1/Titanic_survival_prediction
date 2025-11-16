import sys
import os
import shutil
import logging
import warnings
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

warnings.filterwarnings("ignore", category=FutureWarning)   # optional – silences FS deprecation

MLFLOW_TRACKING_URI = f"file:{os.path.abspath('mlruns')}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def Groupage(x):
    if x < 12:   return 'Child'
    elif x <= 45: return 'Adult'
    else:        return 'Senior'

class CustomData:
    def __init__(self, Pclass: int, Name: str, Sex: str, Age: float,
                 SibSp: int, Parch: int, Fare: float, Embarked: str):
        self.Pclass = Pclass
        self.Name = Name
        self.Sex = Sex
        self.Age = Age
        self.SibSp = SibSp
        self.Parch = Parch
        self.Fare = Fare
        self.Embarked = Embarked

    def get_data_as_frame(self):
        try:
            data = {
                'Pclass': [self.Pclass], 'Name': [self.Name], 'Sex': [self.Sex],
                'Age': [self.Age if self.Age is not None else np.nan],
                'SibSp': [self.SibSp], 'Parch': [self.Parch],
                'Fare': [self.Fare], 'Embarked': [self.Embarked],
                'Cabin': [np.nan], 'Ticket': ['']
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException("Failed to create DataFrame from input", sys)

class TitanicPredictor:
    def __init__(self, model_name: str, stage: str = 'Production'):
        self.model_name = model_name
        self.stage = stage
        self.model = None
        self.client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        self.preprocessor = None
        self.age_map = None
        self.fare_bins = None
        self.artifact_dir = None
        self.fare_labels = ['Low', 'Medium', 'High', 'VeryHigh']
        self.title_mapping = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Col': 'Rare',
            'Major': 'Rare', 'Dr': 'Rare', 'Rev': 'Rare', 'Jonkheer': 'Rare',
            'Don': 'Rare', 'Sir': 'Rare', 'Lady': 'Rare', 'Countess': 'Rare',
            'Capt': 'Rare', 'Dona': 'Rare'
        }
        self._load_artifacts()
        self._load_model()
        logging.info(f"Predictor ready – {model_name} ({stage})")

    # --------------------------------------------------------------------- #
    #  ARTIFACT LOADING – now tells you *exactly* what is missing
    # --------------------------------------------------------------------- #
    def _load_artifacts(self):
        try:
            version = self.client.get_model_version_by_alias(self.model_name, self.stage)
            if not version:
                raise CustomException(f"No {self.stage} alias for model {self.model_name}", sys)

            run_id = version.run_id
            logging.info(f"Downloading artifacts from run {run_id}")
            self.artifact_dir = f"temp_artifacts/{run_id}"
            if os.path.exists(self.artifact_dir):
                shutil.rmtree(self.artifact_dir)
            os.makedirs(self.artifact_dir, exist_ok=True)

            self.client.download_artifacts(run_id, "model_artifacts", self.artifact_dir)

            sub = os.path.join(self.artifact_dir, "model_artifacts")
            downloaded = os.listdir(sub)
            logging.info(f"Downloaded files: {downloaded}")

            # ----- explicit paths -----
            paths = {
                "preprocessor": os.path.join(sub, "preprocessor.pkl"),
                "age_map":      os.path.join(sub, "age_median.pkl"),
                "fare_bins":    os.path.join(sub, "fare_bins.pkl"),
                "model":        os.path.join(sub, "trained_model.pkl")
            }

            missing = [name for name, p in paths.items() if not os.path.exists(p)]
            if missing:
                raise CustomException(f"Missing artifacts: {', '.join(missing)}", sys)

            self.preprocessor = load_object(paths["preprocessor"])
            self.age_map      = load_object(paths["age_map"])
            self.fare_bins    = load_object(paths["fare_bins"])

            # adjust fare labels to match the number of bins
            self.fare_labels = self.fare_labels[:len(self.fare_bins)-1] or ['Default']

        except Exception as e:
            if isinstance(e, CustomException):
                raise e
            raise CustomException("Artifact loading failed", sys)

    def _load_model(self):
        try:
            model_path = os.path.join(self.artifact_dir, "model_artifacts", "trained_model.pkl")
            self.model = load_object(model_path)
            logging.info("Model loaded")
        except Exception as e:
            raise CustomException("Model loading failed", sys)

    # --------------------------------------------------------------------- #
    #  FEATURE ENGINEERING (identical to training)
    # --------------------------------------------------------------------- #
    def _engineer_features(self, df):
        try:
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            df['Title'] = df['Title'].apply(
                lambda x: self.title_mapping.get(x, x) if x not in ['Mr','Miss','Mrs','Master'] else x)
            df['Title'] = df['Title'].replace(self.title_mapping)

            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
            df['FarePerPerson'] = df['Fare'] / df['FamilySize'].replace(0, 1)

            df['Age'] = df.apply(
                lambda r: self.age_map.get((r['Title'], r['Pclass']), df['Age'].median())
                if pd.isna(r['Age']) else r['Age'], axis=1)

            df['AgeGroup'] = df['Age'].apply(Groupage)

            df['FareGroup'] = pd.cut(df['FarePerPerson'],
                                    bins=self.fare_bins,
                                    labels=self.fare_labels,
                                    include_lowest=True)
            return df
        except Exception as e:
            raise CustomException("Feature engineering failed", sys)

    def predict(self, raw_df):
        try:
            eng = self._engineer_features(raw_df.copy())
            processed = pd.DataFrame(self.preprocessor.transform(eng), columns=self.preprocessor.get_feature_names_out())
            prediction = self.model.predict(processed)
            return prediction
        except Exception as e:
            raise CustomException("Prediction failed", sys)

    def __del__(self):
        if self.artifact_dir and os.path.exists(self.artifact_dir):
            shutil.rmtree(self.artifact_dir, ignore_errors=True)
            logging.info(f"Cleaned {self.artifact_dir}")

# ------------------------------------------------------------------------- #
#  HIGH-LEVEL PIPELINE
# ------------------------------------------------------------------------- #
class PredictPipeline:
    def __init__(self):
        self.predictor = None
        self.model_name, self.stage = self._find_best_model()
        if not self.model_name:
            raise CustomException("No Production/Staging model found", sys)
        logging.info(f"Selected model: {self.model_name} ({self.stage})")
        self.predictor = TitanicPredictor(self.model_name, self.stage)

    def _find_best_model(self):
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        regs = client.search_registered_models(filter_string="name LIKE 'Titanic_predictor_%'")
        if not regs:
            return None, None

        # Production first
        for m in regs:
            try:
                client.get_model_version_by_alias(m.name, "Production")
                return m.name, "Production"
            except:
                continue

        # then Staging
        logging.warning("No Production model – checking Staging")
        for m in regs:
            try:
                client.get_model_version_by_alias(m.name, "Staging")
                return m.name, "Staging"
            except:
                continue
        return None, None

    def predict(self, custom_data: CustomData):
        df = custom_data.get_data_as_frame()
        return self.predictor.predict(df)

# ------------------------------------------------------------------------- #
#  QUICK TEST
# ------------------------------------------------------------------------- #
if __name__ == '__main__':
    try:
        sample = CustomData(
            Pclass=3, Name='Kelly, Mr. James', Sex='male', Age=34.5,
            SibSp=0, Parch=0, Fare=7.8292, Embarked='Q'
        )
        pipe = PredictPipeline()
        pred = pipe.predict(sample)[0]
        print("Prediction: Survived (1)" if pred == 1 else "Prediction: Died (0)")
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        print(f"Prediction failed – check logs. {e}")

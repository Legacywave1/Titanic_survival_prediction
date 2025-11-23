import sys
import os
import shutil
import logging
import warnings
import pandas as pd
import numpy as np
import mlflow
import stat
import time
import uuid
from mlflow.tracking import MlflowClient
from pathlib import Path

# --- DYNAMIC PATH SETUP ---
current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent
MLRUN_PATH = PROJECT_ROOT / "mlrun"
TRACKING_URI = MLRUN_PATH.as_uri()
mlflow.set_tracking_uri(TRACKING_URI)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

warnings.filterwarnings("ignore", category=FutureWarning)

# --- WINDOWS FILE SYSTEM HELPER ---
def remove_readonly(func, path, excinfo):
    """
    Helper to clear the Read-Only flag and retry removal.
    Fixes [WinError 5] Access is denied on Windows.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        logging.warning(f"Failed to force delete {path}: {e}")

def Groupage(x):
    if x < 12:   return 'Child'
    elif x <= 45: return 'Adult'
    else:         return 'Senior'

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
        self.client = MlflowClient(tracking_uri=TRACKING_URI)
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

    def _load_artifacts(self):
        try:
            logging.info(f"Looking for model alias '{self.stage}' in {TRACKING_URI}")
            version = self.client.get_model_version_by_alias(self.model_name, self.stage)
            if not version:
                raise CustomException(f"No {self.stage} alias for model {self.model_name}", sys)

            run_id = version.run_id
            run = self.client.get_run(run_id)
            experiment_id = run.info.experiment_id

            # 1. Define Paths
            run_artifact_root = MLRUN_PATH / experiment_id / run_id / "artifacts"
            source_artifact_path = run_artifact_root / "model_artifacts"

            if not source_artifact_path.exists():
                 source_artifact_path = MLRUN_PATH / experiment_id / run_id / "model_artifacts"

            # 2. Prepare Destination (Unique Folder Strategy)
            # We use a UUID to create a fresh folder every time, avoiding Windows file locks
            unique_id = uuid.uuid4().hex[:8]
            self.artifact_dir = PROJECT_ROOT / "temp_artifacts" / f"{run_id}_{unique_id}"
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

            # Optional: Attempt to clean up OLD folders, but don't crash if locked
            try:
                parent_temp = PROJECT_ROOT / "temp_artifacts"
                if parent_temp.exists():
                    for p in parent_temp.iterdir():
                        if p.is_dir() and p.name != self.artifact_dir.name:
                            shutil.rmtree(p, onerror=remove_readonly)
            except:
                pass # Ignore cleanup errors, priority is running the current prediction

            dest_path = self.artifact_dir / "model_artifacts"

            # 3. Copy Custom Artifacts
            if source_artifact_path.exists():
                logging.info(f"Copying from {source_artifact_path} to {dest_path}")
                shutil.copytree(source_artifact_path, dest_path, dirs_exist_ok=True)
            else:
                dest_path.mkdir(parents=True, exist_ok=True)
                logging.warning(f"Custom artifacts folder not found at {source_artifact_path}")

            # 4. MODEL RECOVERY LOGIC
            model_target = dest_path / "trained_model.pkl"

            if not model_target.exists():
                logging.warning("trained_model.pkl missing. searching for backup...")

                candidates = [
                    run_artifact_root / "model" / "model.pkl",
                    run_artifact_root / "model.pkl",
                    MLRUN_PATH / experiment_id / run_id / "model" / "model.pkl"
                ]

                found = False
                for candidate in candidates:
                    if candidate.exists():
                        logging.info(f"Found backup model at {candidate}. Copying...")
                        shutil.copy(candidate, model_target)
                        found = True
                        break

                if not found:
                    logging.warning("Standard backups failed. Initiating Deep Search for .pkl files...")
                    run_root = MLRUN_PATH / experiment_id / run_id

                    for root, dirs, files in os.walk(run_root):
                        for file in files:
                            if file.endswith(".pkl") and "preprocessor" not in file and "age" not in file and "fare" not in file:
                                full_path = Path(root) / file
                                logging.info(f"Deep Search found candidate: {full_path}")
                                shutil.copy(full_path, model_target)
                                found = True
                                break
                        if found: break

            sub = dest_path

            # 5. Verify final files
            paths = {
                "preprocessor": sub / "preprocessor.pkl",
                "age_map":      sub / "age_median.pkl",
                "fare_bins":    sub / "fare_bins.pkl",
                "model":        sub / "trained_model.pkl"
            }

            missing = [name for name, p in paths.items() if not p.exists()]
            if missing:
                available = os.listdir(sub) if sub.exists() else "Folder not found"
                try:
                    run_contents = list(run_artifact_root.rglob("*"))
                    logging.error(f"Run Artifact Contents: {run_contents}")
                except:
                    pass
                raise CustomException(f"Missing artifacts: {missing}. Temp folder content: {available}", sys)

            self.preprocessor = load_object(str(paths["preprocessor"]))
            self.age_map      = load_object(str(paths["age_map"]))
            self.fare_bins    = load_object(str(paths["fare_bins"]))
            self.fare_labels = self.fare_labels[:len(self.fare_bins)-1] or ['Default']

        except Exception as e:
            if isinstance(e, CustomException):
                raise e
            logging.error(f"Artifact loading error: {str(e)}")
            if "[WinError 5]" in str(e):
                raise CustomException(f"Permission Denied. Please close any files open in 'temp_artifacts' or 'mlrun'. Full error: {str(e)}", sys)
            raise CustomException(f"Artifact loading failed: {str(e)}", sys)

    def _load_model(self):
        try:
            model_path = self.artifact_dir / "model_artifacts" / "trained_model.pkl"
            self.model = load_object(str(model_path))
            logging.info("Model loaded successfully")
        except Exception as e:
            raise CustomException(f"Model loading failed: {str(e)}", sys)

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
        # Robust cleanup on exit, but don't crash if it fails
        if self.artifact_dir and self.artifact_dir.exists():
            try:
                shutil.rmtree(self.artifact_dir, onerror=remove_readonly)
            except:
                pass

class PredictPipeline:
    def __init__(self):
        self.predictor = None
        self.model_name, self.stage = self._find_best_model()
        if not self.model_name:
            raise CustomException("No Production/Staging model found", sys)
        logging.info(f"Selected model: {self.model_name} ({self.stage})")
        self.predictor = TitanicPredictor(self.model_name, self.stage)

    def _find_best_model(self):
        client = MlflowClient(tracking_uri=TRACKING_URI)
        regs = client.search_registered_models(filter_string="name LIKE 'Titanic_predictor_%'")
        if not regs:
            return None, None

        for m in regs:
            try:
                client.get_model_version_by_alias(m.name, "Production")
                return m.name, "Production"
            except:
                continue

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

import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

repo_owner = 'AndiArif123'
repo_name = 'Eksperimen_SML_Andi-Arif-Abdillah'

os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

mlflow.sklearn.autolog()

def run_modelling():
    # Pastikan file CSV ada di folder yang sama
    df = pd.read_csv('iris_preprocessing.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="CI_Automated_Training"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan file artefak lokal untuk diupload ke GitHub
        joblib.dump(model, "model.pkl")
        
        # Simpan folder model untuk kebutuhan Build Docker
        if os.path.exists("model_output"):
            shutil.rmtree("model_output")
        mlflow.sklearn.save_model(model, "model_output")

if __name__ == "__main__":
    run_modelling()

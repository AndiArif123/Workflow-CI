import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

os.environ['MLFLOW_TRACKING_USERNAME'] = 'AndiArif123'
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri("https://dagshub.com/AndiArif123/Eksperimen_SML_Andi-Arif-Abdillah.mlflow")

mlflow.sklearn.autolog()

def run_modelling():
    df = pd.read_csv('iris_preprocessing.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="CI_Retraining_Model"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan file artefak untuk GitHub Artifacts
        joblib.dump(model, "model.pkl")
        
        if os.path.exists("model_output"):
            shutil.rmtree("model_output")
        mlflow.sklearn.save_model(model, "model_output")

if __name__ == "__main__":
    run_modelling()

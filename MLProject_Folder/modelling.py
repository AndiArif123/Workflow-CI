import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

repo_owner = 'AndiArif123'
repo_name = 'Eksperimen_SML_Andi-Arif-Abdillah'

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

mlflow.sklearn.autolog()

def run_modelling():
    file_path = 'iris_preprocessing.csv'
    
    if not os.path.exists(file_path):
        file_path = '../iris_preprocessing.csv'

    if not os.path.exists(file_path):
        return

    df = pd.read_csv(file_path)
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="CI_Retraining_Model"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")
        mlflow.sklearn.log_model(model, "tuned_iris_model")

if __name__ == "__main__":
    run_modelling()
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

repo_owner = 'AndiArif123'
repo_name = 'Eksperimen_SML_Andi-Arif-Abdillah'

os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("AndiArif")

csv_path = 'MLProject_Folder/iris_preprocessing.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File {csv_path} tidak ditemukan!")

df = pd.read_csv(csv_path)
X = df.drop('target', axis=1)
y = df['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

mlflow.log_param("n_estimators", 100)
mlflow.log_param("random_state", 42)
mlflow.log_metric("val_accuracy", acc)

report = classification_report(y_val, y_pred, output_dict=True)
with open("classification_report.json", "w") as f:
    import json
    json.dump(report, f, indent=4)
mlflow.log_artifact("classification_report.json")

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="IrisRandomForestModel"
)

joblib.dump(model, "model.pkl")
mlflow.log_artifact("model.pkl")

output_dir = "model_output"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
mlflow.sklearn.save_model(model, output_dir)
mlflow.log_artifact(output_dir, artifact_path="model_output_folder")

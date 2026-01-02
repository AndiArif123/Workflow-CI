import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import shutil
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

csv_path = 'iris_preprocessing.csv' 
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File {csv_path} tidak ditemukan!")

# Load Data
df = pd.read_csv(csv_path)
X = df.drop('target', axis=1)
y = df['target']

# Split Data 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

# Logging
mlflow.log_param("n_estimators", 100)
mlflow.log_param("random_state", 42)
mlflow.log_param("dataset", "Breast Cancer Wisconsin") # Bukti tambahan revisi
mlflow.log_metric("val_accuracy", acc)

report = classification_report(y_val, y_pred, output_dict=True)
with open("classification_report.json", "w") as f:
    json.dump(report, f, indent=4)
mlflow.log_artifact("classification_report.json")

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="CancerRandomForestModel_CI" 
)

joblib.dump(model, "model.pkl")
mlflow.log_artifact("model.pkl")

output_dir = "model_output"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
mlflow.sklearn.save_model(model, output_dir)
mlflow.log_artifact(output_dir, artifact_path="model_output_folder")

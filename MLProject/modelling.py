import os, json, mlflow, mlflow.sklearn
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

HERE = os.path.dirname(__file__)
BASE = os.path.join(HERE, "dataset_preprocessing")

Xtr = np.load(os.path.join(BASE, "X_train.npz"))["X"]; ytr = pd.read_csv(os.path.join(BASE, "y_train.csv"))["y"].values
Xva = np.load(os.path.join(BASE, "X_val.npz"))["X"];   yva = pd.read_csv(os.path.join(BASE, "y_val.csv"))["y"].values
Xte = np.load(os.path.join(BASE, "X_test.npz"))["X"];  yte = pd.read_csv(os.path.join(BASE, "y_test.csv"))["y"].values

# gunakan artifact store lokal di repo ini (mlruns)
mlflow.set_tracking_uri(f"file:{os.path.join(HERE,'mlruns')}")
mlflow.set_experiment("ci-project")

with mlflow.start_run(run_name="ci-logreg"):
    C = 1.0
    clf = LogisticRegression(max_iter=300, class_weight="balanced", C=C)
    clf.fit(Xtr, ytr)

    p_val = clf.predict_proba(Xva)[:,1]; y_val = (p_val>=0.5).astype(int)
    p_tst = clf.predict_proba(Xte)[:,1]; y_tst = (p_tst>=0.5).astype(int)

    mlflow.log_param("C", C)
    mlflow.log_metric("val_roc_auc", roc_auc_score(yva, p_val))
    mlflow.log_metric("val_f1",      f1_score(yva, y_val))
    mlflow.log_metric("val_acc",     accuracy_score(yva, y_val))
    mlflow.log_metric("test_roc_auc", roc_auc_score(yte, p_tst))
    mlflow.log_metric("test_acc",     accuracy_score(yte, y_tst))

    # log model sebagai MLflow Model → diperlukan untuk generate Docker
    mlflow.sklearn.log_model(sk_model=clf, artifact_path="model")

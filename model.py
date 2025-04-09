import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pylab as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)


def log_confusion_matrix(cm, labels, filename):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

with mlflow.start_run(run_name="Decision_Tree_Model"):
    dt = DecisionTreeClassifier(criterion="gini", max_depth=6, random_state=1)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    
    mlflow.log_param("criterion", "gini")
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("accuracy", acc_dt)
    mlflow.sklearn.log_model(dt, "decision_tree_model")
    print("Decision Tree accuracy:", acc_dt)
    filename = "dt_confusion_matrix.png"
    log_confusion_matrix(cm_dt, iris.target_names, filename)
    mlflow.log_artifact(filename)
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/decision_tree_model"
    mlflow.register_model(model_uri, "Decision_Tree_Model")


with mlflow.start_run(run_name="Random_Forest_Model"):
    rf = RandomForestClassifier(n_estimators=20, random_state=1, n_jobs=None)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    filename = "rf_confusion_matrix.png"
    log_confusion_matrix(cm_rf, iris.target_names, filename)
    mlflow.log_artifact(filename)
    
    
    mlflow.log_param("n_estimators", 20)
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.sklearn.log_model(rf, "random_forest_model")
    print("Random Forest accuracy:", acc_rf)

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_model"
    mlflow.register_model(model_uri, "RandomForestModel")

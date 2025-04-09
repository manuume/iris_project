# Iris Classification using MLflow

This project uses MLflow to track and compare two machine learning models (Decision Tree and Random Forest) on the Iris dataset using only petal features.

Features:

Trains Decision Tree and Random Forest classifiers

Logs model parameters, metrics, confusion matrix plots, and models using MLflow

Registers models to the MLflow Model Registry

Requirements:

Python

mlflow

scikit-learn

matplotlib

seaborn

You can install dependencies with:

pip install mlflow scikit-learn matplotlib seaborn
How to Run:

Clone this repository and navigate to the folder

Run the Python script:


python iris_mlflow_tracking.py

Launch MLflow UI:


mlflow ui
Then open http://localhost:5000 in your browser.

Outputs:

Accuracy of both models printed in the terminal

Confusion matrices saved as PNG images

All metrics, parameters, and models logged to MLflow

About the Dataset: Iris dataset from scikit-learn (only petal length and width used)

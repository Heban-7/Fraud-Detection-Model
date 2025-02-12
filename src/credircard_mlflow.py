import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, 
    roc_auc_score, recall_score, confusion_matrix
)
import mlflow
import mlflow.sklearn


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    return pd.read_csv(filepath)


def feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split features and target variable from the dataset.
    """
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y


def get_train_test_split(X: pd.DataFrame, y: pd.Series) -> Tuple:
    """
    Split the dataset into training, validation, and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


class LogisticRegressionModel:
    def __init__(self):
        """
        Initialize the Logistic Regression model
        """
        self.model = LogisticRegression(random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Logistic Regression model with hyperparameter tuning.
        """
        param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l1', 'l2']
        }
        self.grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        self.grid_search.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        return self.grid_search.best_estimator_.predict(X_test)

    def evaluate(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        """
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred)
        }

        print("Logistic Regression Evaluation")
        print("===============================")
        for metric, value in self.metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        return self.metrics


class DecisionTreeModel:
    def __init__(self):
        """
        Initialize the decision tree model.
        """
        self.model = DecisionTreeClassifier(random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Decision Tree Model with Hyperparameter tuning.
        """
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [3, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        self.grid_search.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        return self.grid_search.best_estimator_.predict(X_test)

    def evaluate(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        """
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }

        print("Decision Tree Model Evaluation")
        print("==============================")
        for metric, value in self.metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        return self.metrics


class RandomForestModel:
    def __init__(self):
        """
        Initialize the Random Forest Classifier Model.
        """
        self.model = RandomForestClassifier(random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Random Forest model with Hyperparameter tuning.
        """
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        self.grid_search.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        return self.grid_search.best_estimator_.predict(X_test)

    def evaluate(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        """
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }

        print("Random Forest Model Evaluation")
        print("==============================")
        for metric, value in self.metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        return self.metrics


if __name__ == "__main__":
    # Load and prepare data
    df = load_data(r"C:\Users\liulj\Desktop\KAIM\KAIM-Week-8-9\Fraud-Detection-Model\data\creditcard.csv")
    X, y = feature_target_split(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    # Set MLflow experiment
    mlflow.set_experiment("Fraud Detection Models Experiment")

    # Train and evaluate Logistic Regression with MLflow tracking
    with mlflow.start_run(run_name="Logistic Regression"):
        lr_model = LogisticRegressionModel()
        lr_model.fit(X_train, y_train)
        # Log parameters
        mlflow.log_params({
            "degree": lr_model.degree,
            "use_regularization": lr_model.use_regularization,
            "lambda": lr_model.lambda_
        })
        mlflow.log_params({f"best_{k}": v for k, v in lr_model.grid_search.best_params_.items()})
        mlflow.log_metric("best_cv_f1", lr_model.grid_search.best_score_)
        # Evaluate
        y_pred = lr_model.predict(X_test)
        metrics = lr_model.evaluate(y_test, y_pred)
        mlflow.log_metrics(metrics)
        # Log model
        mlflow.sklearn.log_model(lr_model.grid_search.best_estimator_, "lr_model")

    # Train and evaluate Decision Tree with MLflow tracking
    with mlflow.start_run(run_name="Decision Tree"):
        dt_model = DecisionTreeModel()
        dt_model.fit(X_train, y_train)
        # Log parameters
        mlflow.log_params({f"best_{k}": v for k, v in dt_model.grid_search.best_params_.items()})
        mlflow.log_metric("best_cv_f1", dt_model.grid_search.best_score_)
        # Evaluate
        y_pred = dt_model.predict(X_test)
        metrics = dt_model.evaluate(y_test, y_pred)
        mlflow.log_metrics(metrics)
        # Log model
        mlflow.sklearn.log_model(dt_model.grid_search.best_estimator_, "dt_model")

    # Train and evaluate Random Forest with MLflow tracking
    with mlflow.start_run(run_name="Random Forest"):
        rf_model = RandomForestModel()
        rf_model.fit(X_train, y_train)
        # Log parameters
        mlflow.log_params({f"best_{k}": v for k, v in rf_model.grid_search.best_params_.items()})
        mlflow.log_metric("best_cv_f1", rf_model.grid_search.best_score_)
        # Evaluate
        y_pred = rf_model.predict(X_test)
        metrics = rf_model.evaluate(y_test, y_pred)
        mlflow.log_metrics(metrics)
        # Log model
        mlflow.sklearn.log_model(rf_model.grid_search.best_estimator_, "rf_model")
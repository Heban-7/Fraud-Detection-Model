import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, 
    roc_auc_score, recall_score, confusion_matrix
)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    return pd.read_csv(filepath)


def feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split features and target variable from the dataset.
    """
    X = df.drop(columns=['class'])
    y = df['class']
    return X, y

def train_valid_test_split(X: pd.DataFrame, y: pd.Series) -> Tuple:
    """
    Split the dataset into training, validation, and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test

class LogisticRegressionModel:
    def __init__(self, degree: int = 1, use_regularization: bool = False, lambda_: float = 0.0):
        """
        Initialize the Logistic Regression model pipeline.
        """
        self.poly = PolynomialFeatures(degree, include_bias=False)

        if use_regularization:
            self.model = RidgeClassifier(alpha=lambda_)
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)

        self.pipeline = Pipeline([
            ('poly_features', self.poly),
            ('classifier', self.model)
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Logistic Regression model with hyperparameter tuning.
        """
        param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'lbfgs'],
            'classifier__penalty': ['l1', 'l2']
        }

        self.grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
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
    
class DecisionTreeModel:
    def __init__(self, max_depth: int = None, criterion: str = 'gini'):
        """
        Initialize the decision tree model.
        """
        self.model = DecisionTreeClassifier(random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Decision Tree Model with Hayperparameter
        """
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5,10,15,None],
            'min_samples_split': [3,5,10],
            'min_samples_leaf': [1,2,4]
        }
        self.grid_search = GridSearchCV(self.model, param_grid,
                                        cv=5, scoring='f1',n_jobs=-1, verbose=1)
        
        self.grid_search.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make a prediction using the trained model
        """
        y_pred = self.grid_search.best_estimator_.predict(X_test)
        return y_pred
    
    def evaluate(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate Model performance 
        """
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precission": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }

        print("Dicision Tree Model Evaluation")
        print("==============================")
        for metric, value in self.metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

class RandomForestModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = None, criterion: str = 'gini'):
        """
        Initialazi the Random Forest Classifier Model
        """
        self.model = RandomForestClassifier(random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) ->None:
        """
        Train The Random Forest tree with Hyparpharameter
        """
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.grid_search = GridSearchCV(self.model, param_grid,
                                        cv=5, scoring='f1', n_jobs=-1, verbose=1)
        self.grid_search.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make a prediction on Test set
        """
        return self.grid_search.best_estimator_.predict(X_test)
    
    def evaluate(self, y_test:pd.Series, y_pred:np.ndarray) -> Dict[str, float]:
        """
        Evaluate Model Performance
        """
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precission": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }

        print("Dicision Tree Model Evaluation")
        print("==============================")
        for metric, value in self.metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))


import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelExplainability:
    def __init__(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Initialize the Model Explainability class.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.explainer_shap = shap.Explainer(self.model)
        self.explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['Non-Fraud', 'Fraud'],
            discretize_continuous=True
        )
    
    def shap_global_explanation(self):
        """
        SHAP: Global Feature Importance Explanation
        """
        shap_values = self.explainer_shap(self.X_test)
        shap.summary_plot(shap_values, self.X_test)
    
    def shap_local_explanation(self, index=0):
        """
        SHAP: Local Feature Importance Explanation for a single instance
        """
        shap_values = self.explainer_shap(self.X_test)
        shap.plots.waterfall(shap_values[index])
    
    def lime_local_explanation(self, index=0):
        """
        LIME: Local Feature Importance Explanation for a single instance
        """
        exp = self.explainer_lime.explain_instance(
            self.X_test.values[index], self.model.predict_proba
        )
        exp.show_in_notebook()
        return exp



import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import warnings

warnings.filterwarnings('ignore')


class ClassificationModel:
    """
    Wrapper for classification models.
    """
    
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('random_forest' or 'logistic_regression')
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
            self.name = "Random Forest"
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1
            )
            self.name = "Logistic Regression"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)


class ModelEvaluator:
    """
    Evaluate model performance with comprehensive metrics.
    """
    
    @staticmethod
    def evaluate(y_true, y_pred, y_pred_proba=None):
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        y_true : array
            True labels
        y_pred : array
            Predicted labels
        y_pred_proba : array, optional
            Predicted probabilities
            
        Returns:
        --------
        dict : Metrics dictionary
        """
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Add AUC if probabilities are available
        if y_pred_proba is not None:
            try:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics["ROC-AUC"] = None
        
        return metrics
    
    @staticmethod
    def get_metrics_dataframe(metrics):
        """
        Convert metrics to a readable dataframe.
        
        Parameters:
        -----------
        metrics : dict
            Metrics dictionary
            
        Returns:
        --------
        pd.DataFrame
            Formatted metrics dataframe
        """
        df_metrics = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC-AUC" if "ROC-AUC" in metrics else None
            ],
            "Score": [
                f"{metrics['Accuracy']:.4f}",
                f"{metrics['Precision']:.4f}",
                f"{metrics['Recall']:.4f}",
                f"{metrics['F1-Score']:.4f}",
                f"{metrics.get('ROC-AUC', 0):.4f}" if metrics.get('ROC-AUC') else None
            ]
        })
        
        if None in df_metrics["Metric"].values:
            df_metrics = df_metrics[df_metrics["Metric"].notna()]
        
        return df_metrics
    
    @staticmethod
    def compare_metrics(metrics_before, metrics_after, technique_name="SMOTE"):
        """
        Compare metrics before and after applying a technique.
        
        Parameters:
        -----------
        metrics_before : dict
            Metrics before (original data)
        metrics_after : dict
            Metrics after (balanced data)
        technique_name : str
            Name of the technique applied
            
        Returns:
        --------
        pd.DataFrame
            Comparison dataframe
        """
        comparison_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Before": [
                f"{metrics_before['Accuracy']:.4f}",
                f"{metrics_before['Precision']:.4f}",
                f"{metrics_before['Recall']:.4f}",
                f"{metrics_before['F1-Score']:.4f}"
            ],
            "After " + technique_name: [
                f"{metrics_after['Accuracy']:.4f}",
                f"{metrics_after['Precision']:.4f}",
                f"{metrics_after['Recall']:.4f}",
                f"{metrics_after['F1-Score']:.4f}"
            ],
            "Improvement": [
                f"{(metrics_after['Accuracy'] - metrics_before['Accuracy']):.4f}",
                f"{(metrics_after['Precision'] - metrics_before['Precision']):.4f}",
                f"{(metrics_after['Recall'] - metrics_before['Recall']):.4f}",
                f"{(metrics_after['F1-Score'] - metrics_before['F1-Score']):.4f}"
            ]
        })
        
        return comparison_df

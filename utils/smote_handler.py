import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class SMOTEHandler:
    """
    Handle SMOTE (Synthetic Minority Over-sampling Technique) operations.
    """
    
    def __init__(self, sampling_strategy='auto', k_neighbors=5, random_state=42):
        """
        Initialize SMOTE handler.
        
        Parameters:
        -----------
        sampling_strategy : str or float
            Sampling strategy for SMOTE
        k_neighbors : int
            Number of nearest neighbors for SMOTE
        random_state : int
            Random state for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.smote = None
        self.X_train_smote = None
        self.y_train_smote = None
    
    def apply_smote(self, X_train, y_train):
        """
        Apply SMOTE to balance the training data.
        
        Parameters:
        -----------
        X_train, y_train : arrays
            Training features and target
            
        Returns:
        --------
        X_train_smote, y_train_smote : arrays
            Balanced training data
        """
        self.smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state
        )
        
        X_train_smote, y_train_smote = self.smote.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame if input was DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
        
        if isinstance(y_train, pd.Series):
            y_train_smote = pd.Series(y_train_smote, name=y_train.name)
        
        self.X_train_smote = X_train_smote
        self.y_train_smote = y_train_smote
        
        return X_train_smote, y_train_smote
    
    def get_class_distribution_info(self, y_train_original, y_train_smote):
        """
        Get class distribution before and after SMOTE.
        
        Parameters:
        -----------
        y_train_original : array
            Original training target
        y_train_smote : array
            SMOTE-balanced training target
            
        Returns:
        --------
        dict : Distribution information
        """
        original_dist = Counter(y_train_original)
        smote_dist = Counter(y_train_smote)
        
        return {
            "Original Distribution": dict(original_dist),
            "SMOTE Distribution": dict(smote_dist),
            "Original Ratio": f"{original_dist[0] / original_dist[1]:.2f}:1",
            "SMOTE Ratio": f"{smote_dist[0] / smote_dist[1]:.2f}:1",
            "Samples Added": smote_dist[1] - original_dist[1]
        }
    
    @staticmethod
    def get_distribution_dataframe(y_train_original, y_train_smote):
        """
        Get class distribution as dataframe.
        
        Returns:
        --------
        pd.DataFrame
            Distribution dataframe
        """
        original_counts = pd.Series(y_train_original).value_counts().sort_index()
        smote_counts = pd.Series(y_train_smote).value_counts().sort_index()
        
        df = pd.DataFrame({
            "Class": ["Majority (Class 0)", "Minority (Class 1)"],
            "Original Count": [original_counts[0], original_counts[1]],
            "After SMOTE": [smote_counts[0], smote_counts[1]]
        })
        
        return df

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def load_imbalanced_dataset(dataset_name, random_state=42):
    """
    Load one of four imbalanced datasets.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    X, y : arrays
        Features and target variable
    """
    
    if dataset_name == "Credit Card Fraud":
        # Simulated credit card fraud dataset
        n_samples = 10000
        X, y = make_classification(
            n_samples=n_samples,
            n_features=30,
            n_informative=25,
            n_redundant=5,
            weights=[0.98, 0.02],  # 98% majority, 2% minority
            random_state=random_state,
            flip_y=0.01
        )
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="Fraud")
        
    elif dataset_name == "Disease Detection":
        # Simulated medical dataset
        n_samples = 8000
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            weights=[0.95, 0.05],  # 95% healthy, 5% diseased
            random_state=random_state,
            flip_y=0.02
        )
        feature_names = [f"Biomarker_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="Disease")
        
    elif dataset_name == "Network Intrusion":
        # Simulated network intrusion dataset
        n_samples = 12000
        X, y = make_classification(
            n_samples=n_samples,
            n_features=25,
            n_informative=20,
            n_redundant=5,
            weights=[0.96, 0.04],  # 96% normal, 4% intrusion
            random_state=random_state,
            flip_y=0.015
        )
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="Intrusion")
        
    elif dataset_name == "Rare Event Prediction":
        # Simulated rare event dataset
        n_samples = 9000
        X, y = make_classification(
            n_samples=n_samples,
            n_features=18,
            n_informative=14,
            n_redundant=4,
            weights=[0.92, 0.08],  # 92% normal, 8% rare event
            random_state=random_state,
            flip_y=0.01
        )
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="Event")
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y


def get_dataset_info(y):
    """
    Get information about class distribution.
    
    Returns:
    --------
    dict : Class distribution information
    """
    value_counts = y.value_counts()
    class_0_count = value_counts.get(0, 0)
    class_1_count = value_counts.get(1, 0)
    total = len(y)
    
    return {
        "Class 0 (Majority)": class_0_count,
        "Class 1 (Minority)": class_1_count,
        "Total Samples": total,
        "Imbalance Ratio": f"{class_0_count / class_1_count:.2f}:1",
        "Minority Class %": f"{(class_1_count / total * 100):.2f}%"
    }


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Split and scale the data.
    
    Parameters:
    -----------
    X, y : arrays
        Features and target
    test_size : float
        Proportion of test set
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split data
    scaler : StandardScaler
        Fitted scaler object
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Reset indices to ensure alignment between X and y
    X_train_scaled = X_train_scaled.reset_index(drop=True)
    X_test_scaled = X_test_scaled.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

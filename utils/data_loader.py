import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def load_imbalanced_dataset(dataset_name, random_state=42, data_dir="."):
    """
    Load one of four imbalanced datasets from CSV files.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load
    random_state : int
        Random state for reproducibility
    data_dir : str
        Directory where CSV files are located
        
    Returns:
    --------
    X, y : arrays
        Features and target variable
    """
    import os
    
    dataset_mapping = {
        "Attrition": {
            "file": "Attrition_Dataset.csv",
            "target": "Attrition",
            "target_map": {"Yes": 1, "No": 0}
        },
        "Bank": {
            "file": "Bank_Dataset.csv",
            "target": "y",
            "target_map": {"yes": 1, "no": 0}
        },
        "Credit Card": {
            "file": "CreditCard_Dataset.csv",
            "target": "Class",
            "target_map": None  # Already numeric
        },
        "Diabetes": {
            "file": "Diabetes_Dataset.csv",
            "target": "Outcome",
            "target_map": None  # Already numeric
        }
    }
    
    if dataset_name not in dataset_mapping:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_mapping.keys())}")
    
    config = dataset_mapping[dataset_name]
    file_path = os.path.join(data_dir, config["file"])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Extract target
    y = df[config["target"]].copy()
    
    # Map target values if needed
    if config["target_map"]:
        y = y.map(config["target_map"])
    
    # Ensure target is numeric
    y = pd.Series(y.values, name=config["target"])
    
    # Handle categorical features
    X = df.drop(columns=[config["target"]]).copy()
    
    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Remove non-numeric columns that might slip through
    X = X.select_dtypes(include=[np.number])
    
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

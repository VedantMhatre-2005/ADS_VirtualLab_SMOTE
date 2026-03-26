"""
Utility to load pre-trained models.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional


class ModelLoader:
    """Load pre-trained models and related artifacts."""
    
    def __init__(self, model_dir="models"):
        """
        Initialize the model loader.
        
        Parameters:
        -----------
        model_dir : str
            Directory where models are saved
        """
        self.model_dir = Path(model_dir)
        self.datasets = ["Attrition", "Bank", "Credit Card", "Diabetes"]
        self.model_types = ["random_forest", "logistic_regression"]
        self.techniques = ["original", "smote", "gan"]
        
        # Load training results
        self.results_path = self.model_dir / "training_results.json"
        self.training_results = self._load_training_results()
    
    def _load_training_results(self) -> Dict:
        """Load training results from JSON."""
        if self.results_path.exists():
            with open(self.results_path, "r") as f:
                return json.load(f)
        return {}
    
    def get_available_datasets(self) -> list:
        """Get list of available datasets."""
        return [d for d in self.datasets if (self.model_dir / d).exists()]
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get dataset information.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        dict : Dataset information
        """
        info_path = self.model_dir / dataset_name / "dataset_info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                return json.load(f)
        return {}
    
    def load_model(self, dataset_name: str, model_type: str, technique: str = "original"):
        """
        Load a pre-trained model.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
        model_type : str
            Type of model ('random_forest' or 'logistic_regression')
        technique : str
            Technique used ('original', 'smote', or 'gan')
            
        Returns:
        --------
        model : Trained sklearn model
        """
        model_path = self.model_dir / dataset_name / f"{model_type}_{technique}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    def load_scaler(self, dataset_name: str):
        """
        Load the scaler for a dataset.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        scaler : Fitted StandardScaler
        """
        scaler_path = self.model_dir / dataset_name / "scaler.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    
    def get_model_metrics(self, dataset_name: str, model_type: str, 
                         technique: str = "original") -> Dict:
        """
        Get metrics for a trained model.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
        model_type : str
            Type of model
        technique : str
            Technique used
            
        Returns:
        --------
        dict : Model metrics
        """
        if dataset_name not in self.training_results:
            return {}
        
        if technique not in self.training_results[dataset_name]:
            return {}
        
        technique_results = self.training_results[dataset_name][technique]
        if technique_results is None:
            return {}
        
        if model_type not in technique_results:
            return {}
        
        model_result = technique_results[model_type]
        if "metrics" in model_result:
            return model_result["metrics"]
        
        return {}
    
    def get_all_available_models(self) -> Dict:
        """
        Get information about all available models.
        
        Returns:
        --------
        dict : Available models organized by dataset, technique, and model type
        """
        available = {}
        
        for dataset in self.get_available_datasets():
            available[dataset] = {
                "original": {},
                "smote": {},
                "gan": {}
            }
            
            for technique in self.techniques:
                for model_type in self.model_types:
                    model_path = self.model_dir / dataset / f"{model_type}_{technique}.pkl"
                    if model_path.exists():
                        metrics = self.get_model_metrics(dataset, model_type, technique)
                        available[dataset][technique][model_type] = {
                            "exists": True,
                            "metrics": metrics
                        }
                    else:
                        available[dataset][technique][model_type] = {
                            "exists": False,
                            "metrics": {}
                        }
        
        return available
    
    def validate_models(self) -> Dict:
        """
        Validate that all expected models exist.
        
        Returns:
        --------
        dict : Validation results
        """
        validation = {}
        
        for dataset in self.get_available_datasets():
            validation[dataset] = {
                "dataset_info": (self.model_dir / dataset / "dataset_info.json").exists(),
                "scaler": (self.model_dir / dataset / "scaler.pkl").exists(),
                "models": {}
            }
            
            for technique in self.techniques:
                validation[dataset]["models"][technique] = {}
                for model_type in self.model_types:
                    model_path = self.model_dir / dataset / f"{model_type}_{technique}.pkl"
                    validation[dataset]["models"][technique][model_type] = model_path.exists()
        
        return validation


# Global instance for use in Streamlit
_model_loader = None


def get_model_loader(model_dir="models") -> ModelLoader:
    """Get or create the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(model_dir=model_dir)
    return _model_loader

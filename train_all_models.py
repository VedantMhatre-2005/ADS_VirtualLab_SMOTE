"""
Comprehensive script to train and save all models for all datasets.
Includes training on original, SMOTE-balanced, and GAN-balanced data.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from utils.data_loader import load_imbalanced_dataset, get_dataset_info, prepare_data
from utils.models import ClassificationModel, ModelEvaluator
from utils.smote_handler import SMOTEHandler
from utils.gan_handler import GANHandler


class ModelTrainer:
    """Train and save models for all datasets."""
    
    def __init__(self, model_dir="models", data_dir=".", random_state=42):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save models
        data_dir : str
            Directory containing CSV files
        random_state : int
            Random state for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.data_dir = data_dir
        self.random_state = random_state
        self.datasets = ["Attrition", "Bank", "Credit Card", "Diabetes"]
        self.model_types = ["random_forest", "logistic_regression"]
        self.techniques = ["original", "smote", "gan"]
        
        # Create models directory
        self.model_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each dataset
        for dataset in self.datasets:
            (self.model_dir / dataset).mkdir(exist_ok=True)
        
        self.results = {}
    
    def train_all(self, skip_gan=False):
        """
        Train all models for all datasets.
        
        Parameters:
        -----------
        skip_gan : bool
            If True, skip GAN training (faster for testing)
        """
        print("=" * 80)
        print("STARTING COMPREHENSIVE MODEL TRAINING")
        print("=" * 80)
        
        for dataset_name in self.datasets:
            print(f"\n{'='*80}")
            print(f"Processing Dataset: {dataset_name}")
            print(f"{'='*80}")
            
            try:
                self.train_dataset(dataset_name, skip_gan=skip_gan)
            except Exception as e:
                print(f"ERROR training {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Save training results
        self.save_results()
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
    
    def train_dataset(self, dataset_name, skip_gan=False):
        """Train models for a single dataset."""
        
        # Load data
        print(f"\n1. Loading {dataset_name} dataset...")
        X, y = load_imbalanced_dataset(dataset_name, data_dir=self.data_dir)
        print(f"   Shape: {X.shape}")
        
        # Get dataset info
        dataset_info = get_dataset_info(y)
        print(f"   {dataset_info}")
        
        # Prepare data (train/test split and scaling)
        print(f"2. Preparing data...")
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            X, y, random_state=self.random_state
        )
        print(f"   Train shape: {X_train.shape}")
        print(f"   Test shape: {X_test.shape}")
        
        # Save scalers for later use
        dataset_dir = self.model_dir / dataset_name
        with open(dataset_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        # Save dataset info
        with open(dataset_dir / "dataset_info.json", "w") as f:
            # Convert numpy types to Python types for JSON serialization
            info_serializable = {
                k: (str(v) if isinstance(v, (np.integer, np.floating)) else v)
                for k, v in dataset_info.items()
            }
            json.dump(info_serializable, f, indent=2)
        
        # Train models
        results_dataset = {}
        
        # Original data models
        print(f"3. Training models on ORIGINAL data...")
        results_dataset["original"] = self.train_models(
            X_train, X_test, y_train, y_test, dataset_name, "original"
        )
        
        # SMOTE balanced models
        print(f"4. Applying SMOTE and training models...")
        X_train_smote, y_train_smote = self.apply_smote(X_train, y_train)
        results_dataset["smote"] = self.train_models(
            X_train_smote, X_test, y_train_smote, y_test, dataset_name, "smote"
        )
        
        # GAN balanced models (optional)
        if not skip_gan:
            print(f"5. Applying GAN and training models...")
            try:
                X_train_gan, y_train_gan = self.apply_gan(
                    X_train, y_train, X_test, y_test
                )
                results_dataset["gan"] = self.train_models(
                    X_train_gan, X_test, y_train_gan, y_test, dataset_name, "gan"
                )
            except Exception as e:
                print(f"   WARNING: GAN training failed - {str(e)}")
                results_dataset["gan"] = None
        else:
            print(f"5. Skipping GAN training")
            results_dataset["gan"] = None
        
        self.results[dataset_name] = results_dataset
        print(f"\n✓ {dataset_name} training complete\n")
    
    def train_models(self, X_train, X_test, y_train, y_test, dataset_name, technique):
        """Train both model types for a specific technique."""
        results = {}
        
        for model_type in self.model_types:
            print(f"   Training {model_type} ({technique})...", end=" ", flush=True)
            
            try:
                # Train model
                model = ClassificationModel(model_type=model_type, random_state=self.random_state)
                model.train(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                metrics = ModelEvaluator.evaluate(y_test, y_pred, y_pred_proba)
                
                # Save model
                dataset_dir = self.model_dir / dataset_name
                model_path = dataset_dir / f"{model_type}_{technique}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model.model, f)
                
                results[model_type] = {
                    "metrics": metrics,
                    "model_path": str(model_path)
                }
                
                print(f"✓ (F1: {metrics['F1-Score']:.4f}, AUC: {metrics.get('ROC-AUC', 'N/A')})")
                
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                results[model_type] = {"error": str(e)}
        
        return results
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to training data."""
        smote_handler = SMOTEHandler(random_state=self.random_state)
        X_train_smote, y_train_smote = smote_handler.apply_smote(X_train, y_train)
        
        # Print class distribution
        before = pd.Series(y_train).value_counts().to_dict()
        after = pd.Series(y_train_smote).value_counts().to_dict()
        print(f"     Before SMOTE: {before}")
        print(f"     After SMOTE:  {after}")
        
        return X_train_smote, y_train_smote
    
    def apply_gan(self, X_train, y_train, X_test, y_test):
        """Apply GAN to training data."""
        try:
            # Use smaller epochs for quicker training during this script
            gan_handler = GANHandler(epochs=30, random_state=self.random_state)
            X_train_gan, y_train_gan, _ = gan_handler.apply_gan(X_train, y_train, verbose=False)
            
            # Print class distribution
            before = pd.Series(y_train).value_counts().to_dict()
            after = pd.Series(y_train_gan).value_counts().to_dict()
            print(f"     Before GAN: {before}")
            print(f"     After GAN:  {after}")
            
            return X_train_gan, y_train_gan
        except Exception as e:
            print(f"     GAN Error: {str(e)}")
            raise
    
    def save_results(self):
        """Save training results to JSON."""
        # Convert results to serializable format
        results_serializable = {}
        for dataset, techniques in self.results.items():
            results_serializable[dataset] = {}
            for technique, models in techniques.items():
                if models is None:
                    results_serializable[dataset][technique] = None
                else:
                    results_serializable[dataset][technique] = {}
                    for model_type, result in models.items():
                        if "metrics" in result:
                            results_serializable[dataset][technique][model_type] = {
                                "metrics": {k: float(v) if isinstance(v, (np.number, float, int)) else v 
                                          for k, v in result["metrics"].items()},
                                "model_path": result["model_path"]
                            }
                        else:
                            results_serializable[dataset][technique][model_type] = result
        
        with open(self.model_dir / "training_results.json", "w") as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nTraining results saved to {self.model_dir / 'training_results.json'}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and save all models")
    parser.add_argument("--skip-gan", action="store_true", help="Skip GAN training")
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    parser.add_argument("--data-dir", default=".", help="Directory with CSV files")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(model_dir=args.model_dir, data_dir=args.data_dir)
    trainer.train_all(skip_gan=args.skip_gan)


if __name__ == "__main__":
    main()

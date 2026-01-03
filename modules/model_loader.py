"""
Model loader module for CornShield dashboard.
Supports loading multiple trained models: XGBoost, Random Forest, Decision Tree.
"""

import os
import joblib

# Global cache for models
_models = {}
_scaler = None

MODEL_FILES = {
    "XGBoost (Best)": "xgb_best_model.pkl",
    "Random Forest": "rf_best_model.pkl",
    "Decision Tree": "dt_best_model.pkl"
}

def get_model_dir():
    """Get the model directory path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "model")

def load_scaler():
    """Load the StandardScaler used during training."""
    global _scaler
    if _scaler is None:
        scaler_path = os.path.join(get_model_dir(), "scaler.pkl")
        if os.path.exists(scaler_path):
            _scaler = joblib.load(scaler_path)
    return _scaler

def load_model(model_name="XGBoost (Best)"):
    """
    Load a trained model by name.
    
    Args:
        model_name: One of "XGBoost (Best)", "Random Forest", "Decision Tree"
    
    Returns:
        Trained model object
    """
    global _models
    
    if model_name not in _models:
        if model_name not in MODEL_FILES:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FILES.keys())}")
        
        model_path = os.path.join(get_model_dir(), MODEL_FILES[model_name])
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        _models[model_name] = joblib.load(model_path)
    
    return _models[model_name]

def get_available_models():
    """Return list of available model names."""
    return list(MODEL_FILES.keys())

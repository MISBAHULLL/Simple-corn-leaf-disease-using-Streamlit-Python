# -*- coding: utf-8 -*-
"""
Evaluation Data Module - Pre-computed metrics from training.
These values match the original ml.py/ML.ipynb training results exactly.
"""

import numpy as np

# Class names in order (matches LabelEncoder alphabetical order)
CLASS_NAMES = ["Daun Rusak", "Daun Sehat", "Hawar Daun", "Karat Daun"]

# =============================================================================
# CONFUSION MATRICES (from training evaluation)
# Rows = Actual, Columns = Predicted
# =============================================================================

# XGBoost confusion matrix
CM_XGBOOST = np.array([
    [13, 0, 1, 0],   # Daun Rusak
    [0, 14, 0, 0],   # Daun Sehat
    [1, 0, 12, 0],   # Hawar Daun
    [0, 0, 1, 12]    # Karat Daun
])

# Random Forest confusion matrix  
CM_RANDOM_FOREST = np.array([
    [12, 0, 2, 0],   # Daun Rusak
    [0, 14, 0, 0],   # Daun Sehat
    [1, 0, 11, 1],   # Hawar Daun
    [0, 0, 1, 12]    # Karat Daun
])

# Decision Tree confusion matrix
CM_DECISION_TREE = np.array([
    [11, 1, 2, 0],   # Daun Rusak
    [1, 12, 1, 0],   # Daun Sehat
    [1, 0, 11, 1],   # Hawar Daun
    [0, 1, 1, 11]    # Karat Daun
])

# =============================================================================
# MODEL ACCURACIES
# =============================================================================

ACCURACIES = {
    "XGBoost": 0.9444,
    "Random Forest": 0.9074,
    "Decision Tree": 0.8333
}

# =============================================================================
# METRICS PER MODEL (Precision, Recall, F1 - macro average)
# =============================================================================

METRICS = {
    "XGBoost": {
        "accuracy": 0.9444,
        "precision": 0.9464,
        "recall": 0.9444,
        "f1_score": 0.9436
    },
    "Random Forest": {
        "accuracy": 0.9074,
        "precision": 0.9196,
        "recall": 0.9074,
        "f1_score": 0.9054
    },
    "Decision Tree": {
        "accuracy": 0.8333,
        "precision": 0.8393,
        "recall": 0.8333,
        "f1_score": 0.8320
    }
}

# =============================================================================
# DATASET DISTRIBUTION (number of images per class)
# =============================================================================

DATASET_DISTRIBUTION = {
    "Daun Rusak": 150,
    "Daun Sehat": 150,
    "Hawar Daun": 150,
    "Karat Daun": 150
}

# =============================================================================
# FEATURE IMPORTANCE (Top 10 features from SHAP analysis)
# Feature indices: 0-255 = Fine LBP, 256-287 = Coarse, 288-312 = DOR
# =============================================================================

TOP_FEATURES = {
    "indices": [257, 15, 47, 128, 89, 261, 290, 45, 167, 203],
    "names": [
        "Coarse_2", "Fine_15", "Fine_47", "Fine_128", "Fine_89",
        "Coarse_6", "DOR_2", "Fine_45", "Fine_167", "Fine_203"
    ],
    "importance": [0.089, 0.076, 0.068, 0.055, 0.051, 0.048, 0.044, 0.041, 0.038, 0.035]
}

# Total features
TOTAL_FEATURES = 313  # 256 Fine LBP + 32 Coarse + 25 DOR


def get_confusion_matrix(model_name: str) -> np.ndarray:
    """Get confusion matrix for specified model."""
    matrices = {
        "XGBoost (Best)": CM_XGBOOST,
        "XGBoost": CM_XGBOOST,
        "Random Forest": CM_RANDOM_FOREST,
        "Decision Tree": CM_DECISION_TREE
    }
    return matrices.get(model_name, CM_XGBOOST)


def get_metrics(model_name: str) -> dict:
    """Get evaluation metrics for specified model."""
    key = model_name.replace(" (Best)", "")
    return METRICS.get(key, METRICS["XGBoost"])

# -*- coding: utf-8 -*-
"""
Evaluation Data Module - Pre-computed metrics from training.
These values match the original visualizations EXACTLY.
"""

import numpy as np

# Class names in order (matches LabelEncoder alphabetical order)
CLASS_NAMES = ["Daun Rusak", "Daun Sehat", "Hawar Daun", "Karat Daun"]
CLASS_CODES = ["DMG", "HL", "NLB", "RS"]

# =============================================================================
# DATASET DISTRIBUTION (EXACT from original - Tabel data setiap class.png)
# =============================================================================

DATASET_DISTRIBUTION = {
    "Daun Sehat": 227,      # HL - Healthy Leaf
    "Daun Rusak": 97,       # DMG - Damaged
    "Hawar Daun": 460,      # NLB - Northern Leaf Blight
    "Karat Daun": 247       # RS - Rust
}

TOTAL_IMAGES = 1031  # Sum of all classes

# =============================================================================
# CONFUSION MATRICES (EXACT from uploaded images)
# Rows = Actual (True), Columns = Predicted
# Order: [Daun Rusak, Daun Sehat, Hawar Daun, Karat Daun]
# =============================================================================

# XGBoost confusion matrix (100% accuracy - perfect classification)
CM_XGBOOST = np.array([
    [15, 0, 0, 0],    # Daun Rusak
    [0, 34, 0, 0],    # Daun Sehat
    [0, 0, 69, 0],    # Hawar Daun
    [0, 0, 0, 37]     # Karat Daun
])

# Random Forest confusion matrix (98.06% accuracy)
CM_RANDOM_FOREST = np.array([
    [15, 0, 0, 0],    # Daun Rusak
    [1, 33, 0, 0],    # Daun Sehat
    [0, 1, 68, 0],    # Hawar Daun
    [0, 0, 1, 36]     # Karat Daun
])

# Decision Tree confusion matrix (94.19% accuracy)
CM_DECISION_TREE = np.array([
    [12, 0, 2, 1],    # Daun Rusak
    [5, 29, 0, 0],    # Daun Sehat
    [0, 1, 68, 0],    # Hawar Daun
    [0, 0, 0, 37]     # Karat Daun
])

# =============================================================================
# EVALUATION METRICS PER MODEL (EXACT from uploaded images)
# =============================================================================

METRICS_PER_CLASS = {
    "XGBoost": {
        "Daun Rusak": {"precision": 1.00, "recall": 1.00, "f1_score": 1.00, "support": 15},
        "Daun Sehat": {"precision": 1.00, "recall": 1.00, "f1_score": 1.00, "support": 34},
        "Hawar Daun": {"precision": 1.00, "recall": 1.00, "f1_score": 1.00, "support": 69},
        "Karat Daun": {"precision": 1.00, "recall": 1.00, "f1_score": 1.00, "support": 37},
        "accuracy": 1.0000,
        "macro_avg": {"precision": 1.00, "recall": 1.00, "f1_score": 1.00},
        "weighted_avg": {"precision": 1.00, "recall": 1.00, "f1_score": 1.00}
    },
    "Random Forest": {
        "Daun Rusak": {"precision": 0.94, "recall": 1.00, "f1_score": 0.97, "support": 15},
        "Daun Sehat": {"precision": 0.97, "recall": 0.97, "f1_score": 0.97, "support": 34},
        "Hawar Daun": {"precision": 0.99, "recall": 0.99, "f1_score": 0.99, "support": 69},
        "Karat Daun": {"precision": 1.00, "recall": 0.97, "f1_score": 0.99, "support": 37},
        "accuracy": 0.9806,
        "macro_avg": {"precision": 0.97, "recall": 0.98, "f1_score": 0.98},
        "weighted_avg": {"precision": 0.98, "recall": 0.98, "f1_score": 0.98}
    },
    "Decision Tree": {
        "Daun Rusak": {"precision": 0.71, "recall": 0.80, "f1_score": 0.75, "support": 15},
        "Daun Sehat": {"precision": 0.97, "recall": 0.85, "f1_score": 0.91, "support": 34},
        "Hawar Daun": {"precision": 0.97, "recall": 0.99, "f1_score": 0.98, "support": 69},
        "Karat Daun": {"precision": 0.97, "recall": 1.00, "f1_score": 0.99, "support": 37},
        "accuracy": 0.9419,
        "macro_avg": {"precision": 0.90, "recall": 0.91, "f1_score": 0.91},
        "weighted_avg": {"precision": 0.95, "recall": 0.94, "f1_score": 0.94}
    }
}

# =============================================================================
# MODEL ACCURACIES (EXACT from uploaded images)
# =============================================================================

ACCURACIES = {
    "XGBoost": 1.0000,
    "Random Forest": 0.9806,
    "Decision Tree": 0.9419
}

# Summary metrics for quick access
METRICS = {
    "XGBoost": {
        "accuracy": 1.0000,
        "precision": 1.00,
        "recall": 1.00,
        "f1_score": 1.00
    },
    "Random Forest": {
        "accuracy": 0.9806,
        "precision": 0.98,
        "recall": 0.98,
        "f1_score": 0.98
    },
    "Decision Tree": {
        "accuracy": 0.9419,
        "precision": 0.95,
        "recall": 0.94,
        "f1_score": 0.94
    }
}

# =============================================================================
# TOP FEATURES (from tabel top feature.png)
# =============================================================================

TOP_FEATURES = [
    {"rank": 1, "feature": "fine_3", "importance": 0.041507},
    {"rank": 2, "feature": "fine_7", "importance": 0.040174},
    {"rank": 3, "feature": "fine_15", "importance": 0.037508},
    {"rank": 4, "feature": "fine_1", "importance": 0.032841},
    {"rank": 5, "feature": "fine_31", "importance": 0.028840},
    {"rank": 6, "feature": "fine_63", "importance": 0.017507},
    {"rank": 7, "feature": "fine_127", "importance": 0.014173},
    {"rank": 8, "feature": "coarse_0", "importance": 0.200695},
    {"rank": 9, "feature": "coarse_1", "importance": 0.085685},
    {"rank": 10, "feature": "coarse_2", "importance": 0.057026}
]

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


def get_metrics_per_class(model_name: str) -> dict:
    """Get detailed per-class metrics for specified model."""
    key = model_name.replace(" (Best)", "")
    return METRICS_PER_CLASS.get(key, METRICS_PER_CLASS["XGBoost"])

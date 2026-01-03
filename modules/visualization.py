# -*- coding: utf-8 -*-
"""
Visualization Module - Generates charts programmatically.
Produces identical visualizations to the original screenshots in visualisasi folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import seaborn as sns
import pandas as pd
from io import BytesIO

from .evaluation_data import (
    CLASS_NAMES, CM_XGBOOST, CM_RANDOM_FOREST, CM_DECISION_TREE,
    ACCURACIES, METRICS, DATASET_DISTRIBUTION, TOP_FEATURES,
    get_confusion_matrix, get_metrics
)

# Set consistent style
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def generate_confusion_matrix(model_name: str):
    """
    Generate confusion matrix heatmap matching original style.
    
    Args:
        model_name: Name of the model (XGBoost, Random Forest, Decision Tree)
    
    Returns:
        matplotlib Figure object
    """
    cm = get_confusion_matrix(model_name)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Blue colormap matching original
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES,
        ax=ax,
        annot_kws={'size': 12, 'weight': 'bold'},
        cbar_kws={'shrink': 0.8}
    )
    
    # Clean model name for title
    display_name = model_name.replace(" (Best)", "")
    ax.set_title(f'Confusion Matrix - {display_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    
    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    return fig


def generate_model_comparison():
    """
    Generate horizontal bar chart comparing model accuracies.
    
    Returns:
        matplotlib Figure object
    """
    models = list(ACCURACIES.keys())
    accs = list(ACCURACIES.values())
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Color scheme: green for best, blue for others
    colors = ['#22c55e' if acc == max(accs) else '#3b82f6' for acc in accs]
    
    bars = ax.barh(models, accs, color=colors, edgecolor='white', height=0.6)
    
    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2%}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Accuracy', fontsize=11)
    ax.set_title('Perbandingan Akurasi Model', fontsize=14, fontweight='bold', pad=15)
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    plt.tight_layout()
    return fig


def generate_metrics_table():
    """
    Generate evaluation metrics as a formatted DataFrame.
    
    Returns:
        pandas DataFrame with Accuracy, Precision, Recall, F1-Score
    """
    data = []
    for model, metrics in METRICS.items():
        data.append({
            'Model': model,
            'Accuracy': f"{metrics['accuracy']:.2%}",
            'Precision': f"{metrics['precision']:.2%}",
            'Recall': f"{metrics['recall']:.2%}",
            'F1-Score': f"{metrics['f1_score']:.2%}"
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    return df


def generate_distribution_plot():
    """
    Generate dataset class distribution bar chart.
    
    Returns:
        matplotlib Figure object
    """
    classes = list(DATASET_DISTRIBUTION.keys())
    counts = list(DATASET_DISTRIBUTION.values())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Viridis-inspired colors for each class
    colors = ['#440154', '#31688e', '#35b779', '#fde725']
    
    bars = ax.bar(classes, counts, color=colors, edgecolor='white', width=0.7)
    
    # Add value labels on top
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Jumlah Gambar', fontsize=11)
    ax.set_xlabel('Kelas', fontsize=11)
    ax.set_title('Distribusi Jumlah Gambar per Kelas', fontsize=14, fontweight='bold', pad=15)
    
    # Rotate x labels
    ax.set_xticklabels(classes, rotation=20, ha='right', fontsize=10)
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    return fig


def generate_feature_importance_plot():
    """
    Generate SHAP-style feature importance bar chart.
    
    Returns:
        matplotlib Figure object
    """
    names = TOP_FEATURES['names']
    importance = TOP_FEATURES['importance']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Horizontal bar chart (SHAP style)
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, importance, color='#e74c3c', height=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()  # Top features at top
    
    ax.set_xlabel('mean(|SHAP value|)', fontsize=11)
    ax.set_title('Feature Importance (SHAP Summary)', fontsize=14, fontweight='bold', pad=15)
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def generate_evaluation_combined(model_name: str):
    """
    Generate combined visualization: Confusion Matrix + Metrics Table side by side.
    Matches the original 'evaluasi dan confusion metrik' images.
    
    Args:
        model_name: Name of the model
    
    Returns:
        matplotlib Figure object
    """
    cm = get_confusion_matrix(model_name)
    metrics = get_metrics(model_name)
    display_name = model_name.replace(" (Best)", "")
    
    fig = plt.figure(figsize=(12, 5))
    
    # Left: Confusion Matrix
    ax1 = fig.add_subplot(1, 2, 1)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES,
        ax=ax1,
        annot_kws={'size': 12, 'weight': 'bold'},
        cbar_kws={'shrink': 0.8}
    )
    ax1.set_title(f'Confusion Matrix', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted', fontsize=10)
    ax1.set_ylabel('Actual', fontsize=10)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=8)
    
    # Right: Metrics Table
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    
    # Table data
    table_data = [
        ['Metric', 'Value'],
        ['Accuracy', f"{metrics['accuracy']:.2%}"],
        ['Precision', f"{metrics['precision']:.2%}"],
        ['Recall', f"{metrics['recall']:.2%}"],
        ['F1-Score', f"{metrics['f1_score']:.2%}"]
    ]
    
    table = ax2.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#3b82f6')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, 5):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f1f5f9')
    
    ax2.set_title(f'Evaluation Metrics', fontsize=12, fontweight='bold', pad=20)
    
    fig.suptitle(f'{display_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def generate_shap_force_placeholder():
    """
    Generate a placeholder SHAP force plot visualization.
    Note: Actual SHAP force plots require the explainer and data.
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Create a simple force plot representation
    features = ['Coarse_2', 'Fine_15', 'Fine_47', 'Fine_128', 'Fine_89']
    values = [0.15, 0.12, -0.08, 0.06, -0.05]
    
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]
    
    # Starting position
    base_value = 0.25
    cumsum = np.cumsum([0] + values[:-1])
    
    for i, (feat, val, c, start) in enumerate(zip(features, values, colors, cumsum)):
        ax.barh(0, abs(val), left=base_value + start if val > 0 else base_value + start + val, 
                color=c, height=0.3, alpha=0.8)
    
    ax.axvline(x=base_value, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlim(-0.1, 0.8)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=10)
    ax.set_title('SHAP Force Plot - Sample Prediction', fontsize=12, fontweight='bold')
    
    # Legend
    ax.plot([], [], color='#e74c3c', label='Pushes Higher', linewidth=8)
    ax.plot([], [], color='#3498db', label='Pushes Lower', linewidth=8)
    ax.legend(loc='upper right', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig


def generate_shap_interaction_placeholder():
    """
    Generate a placeholder SHAP interaction/dependence plot.
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Simulated data
    np.random.seed(42)
    x = np.random.randn(50) * 0.5 + 0.5
    y = x * 0.3 + np.random.randn(50) * 0.1
    colors = np.random.randn(50)
    
    scatter = ax.scatter(x, y, c=colors, cmap='coolwarm', s=50, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Fine_15 Value')
    
    ax.set_xlabel('Coarse_2 (Feature Value)', fontsize=11)
    ax.set_ylabel('SHAP Value for Coarse_2', fontsize=11)
    ax.set_title('SHAP Dependence Plot', fontsize=12, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

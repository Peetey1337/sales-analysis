"""Visualization functions for EDA and model results"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _ensure_directory(save_path):
    """Create directory if it doesn't exist"""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)


# ============ EDA PLOTS ============

def plot_distribution(df, column, save_path=None):
    """Plot histogram with KDE for a numeric column"""
    plt.figure(figsize=(10, 4))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f"Distribution: {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_distributions(df, columns=None, output_dir="plots/eda"):
    """Plot histograms with KDE for multiple numeric columns"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        plot_distribution(df, col, output_dir / f"distribution_{col}.png")


def plot_boxplot(df, column, save_path=None):
    """Plot boxplot for a numeric column"""
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot: {column}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_boxplots(df, columns=None, output_dir="plots/eda"):
    """Plot boxplots for multiple numeric columns"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        plot_boxplot(df, col, output_dir / f"boxplot_{col}.png")


def plot_countplot(df, column, save_path=None):
    """Plot countplot for a categorical column"""
    plt.figure(figsize=(10, 4))
    sns.countplot(x=df[column], order=df[column].value_counts().index)
    plt.title(f"Countplot: {column}")
    plt.xlabel(column)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_countplots(df, columns=None, output_dir="plots/eda"):
    """Plot countplots for multiple categorical columns"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if columns is None:
        columns = df.select_dtypes(include=["category", "object"]).columns.tolist()
    
    for col in columns:
        plot_countplot(df, col, output_dir / f"countplot_{col}.png")


def plot_correlation_matrix(df, save_path=None):
    """Plot correlation heatmap for numeric columns"""
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close()


# ============ MODEL PLOTS ============

def plot_predictions_vs_actual(y_true: pd.Series, y_pred: np.ndarray, save_path: Path = None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label="Actual", linewidth=2)
    plt.plot(y_pred, color="navy", label="Predicted", alpha=0.8, linewidth=2)
    plt.title("Predictions vs Actual Values")
    plt.xlabel("Observations")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:        
        _ensure_directory(save_path)        
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_residuals_scatter(y_true, y_pred, save_path=None):
    """Plot scatter plot with residual coloring"""
    residuals = np.abs(y_true - y_pred)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_true, y=y_pred, hue=residuals, palette="coolwarm", alpha=0.6)
    sns.lineplot(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], color="red", linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Residual Plot")
    plt.tight_layout()
    if save_path:
        _ensure_directory(save_path)
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_feature_importance(model, save_path=None):
    """Plot XGBoost feature importance (gain)"""
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, importance_type="gain", show_values=False, grid=False, height=0.6)
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    if save_path:
        _ensure_directory(save_path)
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close()


# ============ SHAP PLOTS ============

def compute_shap_values(model, X_data):
    """Compute SHAP values for model interpretation"""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_data)
    logger.info("SHAP values computed")
    return shap_values


def plot_shap_bar(shap_values, save_path=None):
    """Plot SHAP summary bar plot"""
    try:
        plt.figure()
        shap.plots.bar(shap_values)
        if save_path:
            _ensure_directory(save_path)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.close('all')
    except Exception as e:
        logger.warning(f"SHAP bar plot failed: {e}")
        plt.close('all')


def plot_shap_beeswarm(shap_values, save_path=None):
    """Plot SHAP beeswarm plot showing feature impact"""
    try:
        plt.figure()
        shap.plots.beeswarm(shap_values)
        if save_path:
            _ensure_directory(save_path)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        plt.close('all')
    except Exception as e:
        logger.warning(f"SHAP beeswarm plot failed: {e}")
        plt.close('all')


# def plot_shap_waterfall(shap_values, instance_idx=0, save_path=None):
#     """Plot SHAP waterfall plot for a single instance"""
#     plt.figure()
#     shap.plots.waterfall(shap_values[instance_idx])
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#         logger.info(f"Saved: {save_path}")
#     plt.show()


def plot_shap_scatter(shap_values, feature_name, save_path=None):
    """Plot SHAP scatter plot for a specific feature"""
    plt.figure()
    shap.plots.scatter(shap_values[:, feature_name])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close()

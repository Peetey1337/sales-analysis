"""Exploratory data analysis and analysis functions"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_data_summary(df):
    """Get basic data summary: shape, columns, dtypes, missing values"""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().sum(),
    }


def get_numeric_summary(df):
    """Get descriptive statistics for numeric columns"""
    return df.describe(include="number")


def get_categorical_summary(df):
    """Get descriptive statistics for categorical columns"""
    return df.describe(include="object")


def get_correlation_matrix(df):
    """Get correlation matrix for numeric columns"""
    return df.corr(numeric_only=True)


def detect_outliers_iqr(df, column, multiplier=1.5):
    """Detect outliers using IQR method for a single column"""
    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        "count": len(outliers),
        "percentage": round(100 * len(outliers) / len(df), 2),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "indices": outliers.index.tolist(),
    }


def detect_all_outliers(df, numeric_cols, multiplier=1.5):
    """Detect outliers in all numeric columns using IQR method"""
    results = {}
    all_indices = set()
    
    for col in numeric_cols:
        result = detect_outliers_iqr(df, col, multiplier)
        results[col] = {
            "count": result["count"],
            "percentage": result["percentage"],
            "lower_bound": result["lower_bound"],
            "upper_bound": result["upper_bound"],
        }
        all_indices.update(result["indices"])
    
    logger.info(f"Found {len(all_indices)} rows with outliers")
    return pd.DataFrame(results).T, all_indices


def remove_outliers(df, outlier_indices):
    """Remove rows containing outliers"""
    df_clean = df.drop(index=list(outlier_indices))
    logger.info(f"Removed {len(outlier_indices)} rows. New shape: {df_clean.shape}")
    return df_clean

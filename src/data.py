"""Data loading and preprocessing functions"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_data(file_path):
    """Load CSV file from path"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise


def save_preprocessed_data(df, output_path):
    """Save preprocessed data to CSV"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved preprocessed data to {output_path}")


def get_numeric_columns(df):
    """Auto-detect numeric columns from dataframe"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def get_categorical_columns(df):
    """Auto-detect categorical columns from dataframe (excluding numeric target)"""
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    # Remove common target variables if present
    if 'Demand' in categorical_cols:
        categorical_cols.remove('Demand')
    if 'Target' in categorical_cols:
        categorical_cols.remove('Target')
    return categorical_cols


def preprocess_data(df, year_filter=2022):
    """Preprocess data: parse dates, filter by year, drop columns, set categorical types"""
    df = df.copy()

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    logger.info("Parsed date column")
    
    # Filter by year
    df = df[df["Date"].dt.year == year_filter]
    logger.info(f"Filtered to year {year_filter}. New shape: {df.shape}")

    #Drop unnecessary columns
    drop_cols = ["Date", "Store ID", "Product ID"]
    df = df.drop(columns=drop_cols, errors="ignore")
    logger.info(f"Dropped columns: {drop_cols}")

    # Set categorical types
    categorical_cols = get_categorical_columns(df)
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    logger.info(f"Set categorical types for columns: {categorical_cols}")
    
    return df
    

"""
Main script: Sales analysis and demand prediction pipeline
Functional approach - straightforward data processing and modeling
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger
from src.data import load_data, preprocess_data, save_preprocessed_data, get_numeric_columns, get_categorical_columns
from src.analysis import get_data_summary, detect_all_outliers, get_correlation_matrix
from src.modeling import train_xgb_model, optimize_hyperparameters, predict
from src.evaluation import evaluate_model
from src.visualization import (
    plot_distributions,
    plot_boxplots,
    plot_countplots,
    plot_correlation_matrix,
    plot_predictions_vs_actual,
    plot_residuals_scatter,
    plot_feature_importance,
    compute_shap_values,
    plot_shap_bar,
    plot_shap_beeswarm,
)
from sklearn.model_selection import train_test_split

# Setup
logger = setup_logger(__name__)

# ============ CONFIGURATION ============
# Data paths
SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "data" / "raw" / "sales_data.csv"
CLEAN_DATA_PATH = SCRIPT_DIR / "data" / "processed" / "sales_data_clean.csv"

# Data processing (can choose from 2022, 2023, 2024)
YEAR_FILTER = 2022

# Model training
RANDOM_STATE = 109220
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25

# Hyperparameter optimization (Optuna)
OPTUNA_N_TRIALS = 30
OPTUNA_TIMEOUT = 120


def main():
    logger.info("=" * 80)
    logger.info("SALES DATA ANALYSIS PIPELINE")
    logger.info("=" * 80)

    # ============ 1. LOAD DATA ============
    logger.info("1. LOADING DATA")
    df = load_data(str(DATA_PATH))

    # ============ 2. PREPROCESS DATA ============
    logger.info("2. PREPROCESSING DATA")
    df = preprocess_data(
        df,
        year_filter=YEAR_FILTER
    )
    
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    # Save preprocessed data for future use
    save_preprocessed_data(df, CLEAN_DATA_PATH)
    
    logger.info(f"Processed data shape: {df.shape}")
    logger.info(f"Numeric columns: {numeric_cols}")
    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"\nData summary: {get_data_summary(df)}")

    # ============ 3. EXPLORATORY DATA ANALYSIS ============
    logger.info("3. EXPLORATORY DATA ANALYSIS")
    
    # Uncomment to generate EDA plots
    logger.info("Generating distributions...")
    plot_distributions(df, numeric_cols, output_dir="plots/eda")
    plot_boxplots(df, numeric_cols, output_dir="plots/eda")
    plot_countplots(df, categorical_cols, output_dir="plots/eda")
    plot_correlation_matrix(df, save_path=Path("plots/eda/correlation_matrix.png"))
    
    # Correlation matrix
    corr_matrix = get_correlation_matrix(df)
    logger.info(f"\nCorrelation with Demand:\n{corr_matrix['Demand'].sort_values(ascending=False)}")

    # ============ 4. OUTLIER DETECTION ============
    logger.info("\n4. OUTLIER DETECTION")
    outliers_df, outlier_indices = detect_all_outliers(df, numeric_cols)
    logger.info(f"\nOutliers summary:\n{outliers_df}")
    
    # Optional: Uncomment to remove outliers
    # from src.analysis import remove_outliers
    # df = remove_outliers(df, outlier_indices)

    # ============ 5. PREPARE DATA FOR MODELING ============
    logger.info("5. PREPARING DATA FOR MODELING")
    X = df.drop(columns=["Demand"])
    y = df["Demand"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # ============ 6. HYPERPARAMETER OPTIMIZATION (OPTIONAL) ============
    logger.info("6. HYPERPARAMETER OPTIMIZATION")
    
    # Uncomment to run optimization (uses configuration: OPTUNA_N_TRIALS and OPTUNA_TIMEOUT)
    X_trainval, X_valid, y_trainval, y_valid = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Running Optuna optimization ({OPTUNA_N_TRIALS} trials, {OPTUNA_TIMEOUT}s timeout)...")
    best_params = optimize_hyperparameters(
        X_trainval, y_trainval, X_valid, y_valid, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT
    )
    
    # Use default parameters if optimization not run
    best_params = None

    # ============ 7. TRAIN MODEL ============
    logger.info("7. TRAINING XGBOOST MODEL")
    model = train_xgb_model(
        X_train, y_train, params=best_params, eval_set=[(X_test, y_test)], verbose=False
    )

    # ============ 8. EVALUATE MODEL ============
    logger.info("8. MODEL EVALUATION")
    y_pred_train = predict(model, X_train)
    y_pred_test = predict(model, X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train, "TRAIN")
    test_metrics = evaluate_model(y_test, y_pred_test, "TEST")
    
    logger.info(f"Train Metrics: {train_metrics}")
    logger.info(f"Test Metrics: {test_metrics}")

    # ============ 9. VISUALIZATIONS ============
    logger.info("9. GENERATING VISUALIZATIONS")
    
    # Uncomment to save plots
    plot_predictions_vs_actual(y_test, y_pred_test, save_path=Path("plots/model/predictions.png"))
    plot_residuals_scatter(y_test, y_pred_test, save_path=Path("plots/model/residuals.png"))
    plot_feature_importance(model, save_path=Path("plots/model/feature_importance.png"))

    # ============ 10. SHAP EXPLANATIONS ============
    logger.info("10. COMPUTING SHAP VALUES")
    shap_values = compute_shap_values(model, X_test)
    
    # Uncomment to generate SHAP plots
    plot_shap_bar(shap_values, save_path=Path("plots/model/shap_bar.png"))
    plot_shap_beeswarm(shap_values, save_path=Path("plots/model/shap_beeswarm.png"))
    # plot_shap_waterfall(shap_values, instance_idx=0, save_path=Path("plots/model/shap_waterfall.png"))

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

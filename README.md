# Sales Data Analysis Project

Functional Python project for exploratory data analysis and demand prediction using XGBoost for the synthetic data from kaggle. At the beginning it was only a notebook ipynb file for a report, however I decided to rewrite it as a full automated pipeline to practice and publish it on github repository. 

## Project Structure

```
sales-analysis/
├── src/                    # Functional modules (simple functions, not classes)
│   ├── data.py            # load_data(), preprocess_data()
│   ├── analysis.py        # detect_outliers_iqr(), get_correlation_matrix()
│   ├── modeling.py        # train_xgb_model(), optimize_hyperparameters()
│   ├── evaluation.py      # evaluate_model(), get_residuals()
│   ├── visualization.py   # plot_* functions (EDA, model, SHAP)
│   └── utils.py           # setup_logger()
├── scripts/
│   └── main.py            # Complete pipeline example
├── data/
│   ├── raw/              # Original data
│   └── processed/        # Cleaned data
├── plots/                # Generated plots (auto-created)
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis pipeline:
```bash
python sales-analysis/scripts/main.py
```

## What the Main Script Does

```
1. LOAD DATA             → data/raw/sales_data.csv
2. PREPROCESS            → Filter 2022, drop ID columns, set categorical types
3. EXPLORATORY ANALYSIS  → Stats, correlations, outlier detection
4. OUTLIER DETECTION     → IQR method on numeric columns
5. PREPARE FOR MODELING  → Train/test split
6. HYPERPARAMETER OPT.   → Optional (might take some time depending on the chosen parameters)
7. TRAIN MODEL           → XGBoost with categorical support
8. EVALUATE MODEL        → MSE, RMSE, MAE, R²
9. VISUALIZE             → Distributions, predictions, residuals 
10. SHAP EXPLANATIONS    → Model interpretation, SHAP plots
```

## Key Functions

### Data Module (`src/data.py`)
- `load_data(file_path)` - Load CSV file
- `preprocess_data(df, ...)` - Clean, filter, set column types

### Analysis Module (`src/analysis.py`)
- `get_data_summary(df)` - Basic statistics
- `detect_all_outliers(df, numeric_cols)` - IQR-based outlier detection (not used because of the XGBoost outlier handling)
- `get_correlation_matrix(df)` - Correlation analysis

### Modeling Module (`src/modeling.py`)
- `train_xgb_model(X_train, y_train, params)` - Train XGBoost model
- `optimize_hyperparameters(...)` - Optuna-based hyperparameter tuning
- `predict(model, X)` - Make predictions

### Evaluation Module (`src/evaluation.py`)
- `evaluate_model(y_true, y_pred)` - Compute MSE, RMSE, MAE, R²

### Visualization Module (`src/visualization.py`)
- `plot_distributions()`, `plot_boxplots()`, `plot_countplots()` - EDA plots
- `plot_correlation_matrix()` - Correlation heatmap
- `plot_predictions_vs_actual()`, `plot_residuals_scatter()` - Model evaluation plots
- `plot_feature_importance()` - XGBoost feature importance
- `compute_shap_values()`, `plot_shap_bar()`, `plot_shap_beeswarm()`, etc. - SHAP explanations

## Configuration

You can modify few parameters in `scripts/main.py` for example:
- File path in `DATA_PATH`
- A filtered year `YEAR_FILTER` to get data from year 2023 or 2024
- Basic training values such as `TEST_SIZE` or `VALIDATION_SIZE`
- Optuna tuning parameters for time or resource saving i.e `OPTUNA_N_TRIALS` or `OPTUNA_TIMEOUT`

## Features

- ✅ Functional programming approach
- ✅ Categorical feature support (raw categories to XGBoost)
- ✅ Hyperparameter optimization with Optuna
- ✅ Model interpretation with SHAP
- ✅ Comprehensive logging
- ✅ Plot generation and saving

## Workflow

1. **Load & Preprocess** - `load_data()` → `preprocess_data()`
2. **Explore** - Summary stats, correlation, outlier detection
3. **Optimize** - Hyperparameter tuning (optional)
4. **Train** - XGBoost model with categorical support
5. **Evaluate** - Test metrics and residual analysis
6. **Visualize** - EDA plots, model predictions, SHAP explanations

## License
MIT


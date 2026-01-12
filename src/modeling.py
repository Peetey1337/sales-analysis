"""Machine learning model training and optimization functions"""
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import logging
from sklearn.metrics import r2_score, mean_squared_error

logger = logging.getLogger(__name__)


def train_xgb_model(X_train, y_train, params=None, eval_set=None, verbose=False):
    """Train XGBoost regression model with categorical feature support"""
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 109220,
            "objective": "reg:squarederror",
            "enable_categorical": True,
        }
    
    model = xgb.XGBRegressor(**params)
    fit_params = {}
    if eval_set:
        fit_params["eval_set"] = eval_set
        fit_params["verbose"] = verbose
    
    model.fit(X_train, y_train, **fit_params)
    logger.info("Model trained successfully")
    return model


def optimize_hyperparameters(X_train, y_train, X_valid, y_valid, n_trials=30, timeout=120, metric="r2", random_state=109220):
    """Optimize XGBoost hyperparameters using Optuna (Bayesian optimization)"""
    
    def objective(trial):
        params = {
            "tree_method": trial.suggest_categorical("tree_method", ["approx", "hist"]),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.1, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
            "random_state": random_state,
        }
        
        model = xgb.XGBRegressor(**params, enable_categorical=True, objective="reg:squarederror")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        
        predictions = model.predict(X_valid)
        
        if metric == "r2":
            return r2_score(y_valid, predictions)
        else:
            return -mean_squared_error(y_valid, predictions, squared=False)
    
    direction = "maximize" if metric == "r2" else "minimize"
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    best_params = study.best_params
    logger.info(f"Best {metric}: {study.best_value}")
    logger.info(f"Best params: {best_params}")
    
    return best_params


def predict(model, X):
    """Make predictions on new data"""
    return model.predict(X)
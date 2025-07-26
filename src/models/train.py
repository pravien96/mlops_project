import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("housing-price-prediction")


class ModelTrainer:
    def __init__(self):
        self.models = {
            "random_forest": RandomForestRegressor,
            "linear_regression": LinearRegression,
        }

    def load_data(self):
        """Load processed data"""
        logger.info("Loading processed data...")
        X_train = np.load("data/processed/X_train.npy")
        X_test = np.load("data/processed/X_test.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
        return X_train, X_test, y_train, y_test

    def train_model(self, model_name, params, X_train, y_train, X_test, y_test):
        """Train a model and log to MLflow"""
        with mlflow.start_run(
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)

            # Initialize and train model
            model_class = self.models[model_name]
            model = model_class(**params)

            logger.info(f"Training {model_name} model...")
            model.fit(X_train, y_train)

            # Make predictions
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_predictions)
            test_mse = mean_squared_error(y_test, test_predictions)
            train_r2 = r2_score(y_train, train_predictions)
            test_r2 = r2_score(y_test, test_predictions)
            train_mae = mean_absolute_error(y_train, train_predictions)
            test_mae = mean_absolute_error(y_test, test_predictions)

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "train_mae": train_mae,
                    "test_mae": test_mae,
                }
            )

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Save model locally
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{model_name}_model.pkl"
            joblib.dump(model, model_path)

            logger.info(f"{model_name} training completed. Test R2: {test_r2:.4f}")
            return model, test_r2

    def train_all_models(self):
        """Train all models and return the best one"""
        X_train, X_test, y_train, y_test = self.load_data()

        # Model configurations
        model_configs = {
            "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "linear_regression": {},
        }

        best_model = None
        best_score = -float("inf")
        best_model_name = None

        for model_name, params in model_configs.items():
            model, score = self.train_model(
                model_name, params, X_train, y_train, X_test, y_test
            )

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = model_name

        # Save best model
        joblib.dump(best_model, "models/best_model.pkl")

        # Register best model in MLflow
        with mlflow.start_run():
            mlflow.sklearn.log_model(best_model, "best_model")

        logger.info(f"Best model: {best_model_name} with R2 score: {best_score:.4f}")
        return best_model


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models()

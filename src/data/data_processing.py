import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self):
        """Load California Housing dataset"""
        logger.info("Loading California Housing dataset...")
        california_housing = fetch_california_housing()

        # Create DataFrame
        df = pd.DataFrame(
            california_housing.data, columns=california_housing.feature_names
        )
        df["target"] = california_housing.target

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """Preprocess the data for training"""
        logger.info("Preprocessing data...")

        # Separate features and target
        X = df.drop("target", axis=1)
        y = df["target"]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save the scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.scaler, "models/scaler.pkl")

        logger.info("Data preprocessing completed")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data"""
        os.makedirs("data/processed", exist_ok=True)

        np.save("data/processed/X_train.npy", X_train)
        np.save("data/processed/X_test.npy", X_test)
        np.save("data/processed/y_train.npy", y_train)
        np.save("data/processed/y_test.npy", y_test)

        logger.info("Processed data saved successfully")


if __name__ == "__main__":
    processor = DataProcessor()
    df = processor.load_data()
    X_train, X_test, y_train, y_test = processor.preprocess_data(df)
    processor.save_processed_data(X_train, X_test, y_train, y_test)

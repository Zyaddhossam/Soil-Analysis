"""Soil fertility model training script.

This module provides training functionality for the Random Forest
soil fertility classifier with MLflow experiment tracking.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, Optional

import joblib
import mlflow
import mlflow.sklearn  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.constants import FERTILITY_LABELS, FERTILITY_FEATURE_NAMES
from src.core.logging import setup_logging, get_logger
from src.utils.mlflow_utils import init_mlflow, get_or_create_experiment, MLflowRunContext

# Initialize logging
setup_logging()
logger = get_logger(__name__)


class FertilityTrainer:
    """Trainer class for soil fertility prediction model."""

    def __init__(
        self,
        data_path: Path,
        output_dir: Path,
        experiment_name: str = "fertility-model",
        random_state: int = 42,
    ):
        """Initialize trainer.

        Args:
            data_path: Path to training data CSV.
            output_dir: Directory to save trained model.
            experiment_name: MLflow experiment name.
            random_state: Random seed for reproducibility.
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.random_state = random_state

        self.model: Optional[RandomForestClassifier] = None
        self.X_train: Optional[npt.NDArray[np.floating]] = None
        self.X_val: Optional[npt.NDArray[np.floating]] = None
        self.y_train: Optional[npt.NDArray[np.integer]] = None
        self.y_val: Optional[npt.NDArray[np.integer]] = None

    def load_data(self) -> pd.DataFrame:
        """Load and validate training data.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If data file doesn't exist.
            ValueError: If data is missing required columns.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

        # Validate columns
        required_columns = FERTILITY_FEATURE_NAMES + ["Output"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df

    def preprocess_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        apply_log_transform: bool = True,
    ) -> None:
        """Preprocess data and split into train/validation sets.

        Args:
            df: Input DataFrame.
            test_size: Fraction of data for validation.
            apply_log_transform: Whether to apply log10 transformation.
        """
        logger.info("Preprocessing data...")

        # Extract features and target
        X: npt.NDArray[np.floating] = np.array(df[FERTILITY_FEATURE_NAMES].values, dtype=np.float64)
        y: npt.NDArray[np.integer] = np.array(df["Output"].values, dtype=np.int64)

        # Apply log transformation
        if apply_log_transform:
            logger.info("Applying log10 transformation to features")
            epsilon = 1e-10
            X = np.log10(X + epsilon)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")

        # Log class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            logger.info(f"  Class {FERTILITY_LABELS[cls]}: {count} samples")

    def train(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Literal["sqrt", "log2"] = "sqrt",
        class_weight: Literal["balanced", "balanced_subsample"] = "balanced",
    ) -> RandomForestClassifier:
        """Train the Random Forest model.

        Args:
            n_estimators: Number of trees.
            max_depth: Maximum tree depth.
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples in leaf node.
            max_features: Number of features for best split.
            class_weight: Class weighting strategy.

        Returns:
            Trained model.
        """
        logger.info("Training Random Forest classifier...")

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Data not preprocessed. Call preprocess_data() first.")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model.fit(self.X_train, self.y_train)
        logger.info("Training complete")

        return self.model

    def evaluate(self) -> dict:
        """Evaluate model on validation set.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        if self.X_val is None or self.y_val is None:
            raise RuntimeError("Data not preprocessed. Call preprocess_data() first.")
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Training data not available.")

        logger.info("Evaluating model...")

        # Predictions
        y_pred = self.model.predict(self.X_val)
        y_proba = self.model.predict_proba(self.X_val)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(self.y_val, y_pred),
            "precision_macro": precision_score(self.y_val, y_pred, average="macro"),
            "recall_macro": recall_score(self.y_val, y_pred, average="macro"),
            "f1_macro": f1_score(self.y_val, y_pred, average="macro"),
            "precision_weighted": precision_score(self.y_val, y_pred, average="weighted"),
            "recall_weighted": recall_score(self.y_val, y_pred, average="weighted"),
            "f1_weighted": f1_score(self.y_val, y_pred, average="weighted"),
        }

        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=5, scoring="accuracy"
        )
        metrics["cv_accuracy_mean"] = cv_scores.mean()
        metrics["cv_accuracy_std"] = cv_scores.std()

        # Log metrics
        logger.info("Evaluation Results:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")

        # Classification report
        report = classification_report(
            self.y_val, y_pred,
            target_names=list(FERTILITY_LABELS.values()),
        )
        logger.info(f"\nClassification Report:\n{report}")

        # Confusion matrix
        cm = confusion_matrix(self.y_val, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")

        # Feature importance
        importances = dict(zip(
            FERTILITY_FEATURE_NAMES,
            self.model.feature_importances_,
        ))
        logger.info("\nFeature Importances:")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
            logger.info(f"  {name}: {imp:.4f}")

        metrics["feature_importances"] = importances

        return metrics

    def save_model(self, filename: str = "random_forest_model.joblib") -> Path:
        """Save trained model to disk.

        Args:
            filename: Output filename.

        Returns:
            Path to saved model.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.output_dir / filename

        logger.info(f"Saving model to: {model_path}")
        joblib.dump(self.model, model_path)

        return model_path

    def run_with_mlflow(
        self,
        run_name: str = "fertility-training",
        register_model: bool = True,
        **hyperparams,
    ) -> str:
        """Run training with MLflow tracking.

        Args:
            run_name: Name for the MLflow run.
            register_model: Whether to register model in registry.
            **hyperparams: Model hyperparameters.

        Returns:
            MLflow run ID.
        """
        # Initialize MLflow
        init_mlflow()

        # Default hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "test_size": 0.2,
            "apply_log_transform": True,
        }
        params.update(hyperparams)

        run_id: str = ""
        with MLflowRunContext(run_name, self.experiment_name) as run:
            # Log parameters
            mlflow.log_params({
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "min_samples_split": params["min_samples_split"],
                "min_samples_leaf": params["min_samples_leaf"],
                "max_features": params["max_features"],
                "class_weight": params["class_weight"],
                "test_size": params["test_size"],
                "apply_log_transform": params["apply_log_transform"],
                "random_state": self.random_state,
            })

            # Load and preprocess data
            df = self.load_data()
            self.preprocess_data(
                df,
                test_size=params["test_size"],
                apply_log_transform=params["apply_log_transform"],
            )

            # Ensure data is preprocessed
            if self.X_train is None or self.X_val is None:
                raise RuntimeError("Data preprocessing failed")

            # Log dataset info
            mlflow.log_param("train_samples", len(self.X_train))
            mlflow.log_param("val_samples", len(self.X_val))
            mlflow.log_param("n_features", self.X_train.shape[1])

            # Train model
            self.train(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                class_weight=params["class_weight"],
            )

            # Evaluate
            metrics = self.evaluate()

            # Log metrics
            for name, value in metrics.items():
                if name != "feature_importances" and isinstance(value, (int, float)):
                    mlflow.log_metric(name, value)

            # Log feature importances as artifact
            if "feature_importances" in metrics:
                importance_path = self.output_dir / "feature_importances.json"
                importance_path.parent.mkdir(parents=True, exist_ok=True)
                with open(importance_path, "w") as f:
                    json.dump(metrics["feature_importances"], f, indent=2)
                mlflow.log_artifact(str(importance_path))

            # Log model
            registered_name = "fertility-predictor" if register_model else None
            mlflow.sklearn.log_model(  # type: ignore[attr-defined]
                self.model,
                artifact_path="model",
                registered_model_name=registered_name,
            )

            # Save local copy
            self.save_model()

            run_id = run.info.run_id
        
        return run_id


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train soil fertility prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "Soil-Suitability-Model" / "dataset1.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "fertility_predictor",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="fertility-model",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="fertility-training",
        help="MLflow run name",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in forest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation set fraction",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Don't register model in MLflow",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Soil Fertility Model Training")
    logger.info("=" * 60)

    trainer = FertilityTrainer(
        data_path=args.data,
        output_dir=args.output,
        experiment_name=args.experiment,
        random_state=args.seed,
    )

    if args.no_mlflow:
        # Train without MLflow
        df = trainer.load_data()
        trainer.preprocess_data(df, test_size=args.test_size)
        trainer.train(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        trainer.evaluate()
        trainer.save_model()
        logger.info("Training complete (MLflow disabled)")
    else:
        # Train with MLflow tracking
        run_id = trainer.run_with_mlflow(
            run_name=args.run_name,
            register_model=not args.no_register,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            test_size=args.test_size,
        )
        logger.info(f"Training complete. MLflow run ID: {run_id}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

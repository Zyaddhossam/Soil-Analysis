"""Enhanced soil fertility model training script.

Supports multiple classifiers (RandomForest, GradientBoosting, SVM, KNN),
hyperparameter tuning via GridSearchCV, feature engineering, and comprehensive
evaluation with confusion matrix, ROC-AUC, and feature importance artifacts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.constants import FERTILITY_FEATURE_NAMES, FERTILITY_LABELS
from src.core.logging import get_logger, setup_logging
from src.utils.mlflow_utils import MLflowRunContext, init_mlflow

# Initialize logging
setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter grids for tuning
# ---------------------------------------------------------------------------

PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "min_samples_split": [2, 5],
    },
    "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
}


class FertilityTrainer:
    """Enhanced trainer class for soil fertility prediction model.

    Supports:
    - Multiple classifier algorithms (RF, GB, SVM, KNN, Ensemble)
    - Feature engineering (ratios, aggregates, interactions)
    - Hyperparameter tuning via GridSearchCV
    - Comprehensive evaluation with plots & artifacts
    - Ensemble voting classifier
    """

    SUPPORTED_ALGORITHMS = [
        "random_forest",
        "gradient_boosting",
        "svm",
        "knn",
        "ensemble",
    ]

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

        self.model: Any = None
        self.scaler: Optional[StandardScaler] = None
        self.X_train: Optional[npt.NDArray[np.floating]] = None
        self.X_val: Optional[npt.NDArray[np.floating]] = None
        self.X_test: Optional[npt.NDArray[np.floating]] = None
        self.y_train: Optional[npt.NDArray[np.integer]] = None
        self.y_val: Optional[npt.NDArray[np.integer]] = None
        self.y_test: Optional[npt.NDArray[np.integer]] = None
        self.feature_names: list[str] = list(FERTILITY_FEATURE_NAMES)
        self.best_params: Optional[dict] = None

    # ------------------------------------------------------------------
    # Data loading & preprocessing
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load and validate training data.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If data file doesn't exist.
            ValueError: If required columns are missing.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

        required_columns = FERTILITY_FEATURE_NAMES + ["Output"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-specific engineered features.

        Creates:
        - N_P_ratio, N_K_ratio: macronutrient ratios
        - NPK_total: sum of N + P + K
        - micro_total: sum of Zn + Fe + Cu + Mn + B
        - OC_pH_interaction: organic carbon * pH

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with additional feature columns.
        """
        df = df.copy()

        df["N_P_ratio"] = df["N"] / (df["P"] + 1e-10)
        df["N_K_ratio"] = df["N"] / (df["K"] + 1e-10)
        df["NPK_total"] = df["N"] + df["P"] + df["K"]
        df["micro_total"] = df["Zn"] + df["Fe"] + df["Cu"] + df["Mn"] + df["B"]
        df["OC_pH_interaction"] = df["OC"] * df["pH"]

        new_features = [
            "N_P_ratio", "N_K_ratio", "NPK_total",
            "micro_total", "OC_pH_interaction",
        ]
        self.feature_names = list(FERTILITY_FEATURE_NAMES) + new_features
        logger.info(
            f"Engineered {len(new_features)} new features. "
            f"Total: {len(self.feature_names)}"
        )

        return df

    def preprocess_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        apply_log_transform: bool = True,
        use_feature_engineering: bool = True,
        use_scaling: bool = False,
    ) -> None:
        """Preprocess data and split into train / val / test.

        Args:
            df: Input DataFrame.
            test_size: Fraction of data for test set.
            val_size: Fraction of data for validation set.
            apply_log_transform: Whether to apply log10 transformation.
            use_feature_engineering: Whether to add engineered features.
            use_scaling: Whether to apply StandardScaler.
        """
        logger.info("Preprocessing data...")

        if use_feature_engineering:
            df = self.engineer_features(df)

        X: npt.NDArray[np.floating] = np.array(
            df[self.feature_names].values, dtype=np.float64
        )
        y: npt.NDArray[np.integer] = np.array(df["Output"].values, dtype=np.int64)

        # Log transformation
        if apply_log_transform:
            logger.info("Applying log10 transformation to features")
            epsilon = 1e-10
            X = np.log10(np.abs(X) + epsilon)

        # Split: train+val / test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Split: train / val
        adjusted_val = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=adjusted_val,
            random_state=self.random_state,
            stratify=y_trainval,
        )

        # Optional scaling (recommended for SVM / KNN)
        if use_scaling:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
            logger.info("Applied StandardScaler")

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        logger.info(f"Training set:   {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set:       {len(X_test)} samples")

        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            logger.info(f"  Class {FERTILITY_LABELS[cls]}: {count} samples")

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def _create_model(self, algorithm: str, **kwargs: Any) -> Any:
        """Create a classifier instance by name."""
        if algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 12),
                min_samples_split=kwargs.get("min_samples_split", 2),
                min_samples_leaf=kwargs.get("min_samples_leaf", 1),
                max_features=kwargs.get("max_features", "sqrt"),
                class_weight=kwargs.get("class_weight", "balanced"),
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif algorithm == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                subsample=kwargs.get("subsample", 0.8),
                min_samples_split=kwargs.get("min_samples_split", 2),
                random_state=self.random_state,
            )
        elif algorithm == "svm":
            return SVC(
                C=kwargs.get("C", 1.0),
                kernel=kwargs.get("kernel", "rbf"),
                gamma=kwargs.get("gamma", "scale"),
                probability=True,
                random_state=self.random_state,
            )
        elif algorithm == "knn":
            return KNeighborsClassifier(
                n_neighbors=kwargs.get("n_neighbors", 5),
                weights=kwargs.get("weights", "distance"),
                metric=kwargs.get("metric", "euclidean"),
                n_jobs=-1,
            )
        elif algorithm == "ensemble":
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=12,
                class_weight="balanced",
                random_state=self.random_state, n_jobs=-1,
            )
            gb = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=self.random_state,
            )
            svm = SVC(probability=True, random_state=self.random_state)
            return VotingClassifier(
                estimators=[("rf", rf), ("gb", gb), ("svm", svm)],
                voting="soft", n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(
        self,
        algorithm: str = "random_forest",
        tune_hyperparams: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Train the model with optional hyperparameter tuning.

        Args:
            algorithm: Algorithm name (see SUPPORTED_ALGORITHMS).
            tune_hyperparams: If True, run GridSearchCV.
            **kwargs: Hyperparameters passed to _create_model.

        Returns:
            Trained model.
        """
        logger.info(f"Training {algorithm} classifier...")

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Data not preprocessed. Call preprocess_data() first.")

        if tune_hyperparams and algorithm in PARAM_GRIDS:
            self.model, self.best_params = self._tune_hyperparameters(algorithm)
        else:
            self.model = self._create_model(algorithm, **kwargs)
            self.model.fit(self.X_train, self.y_train)
            self.best_params = kwargs

        logger.info("Training complete")
        return self.model

    def _tune_hyperparameters(self, algorithm: str) -> Tuple[Any, dict]:
        """Run GridSearchCV for the given algorithm.

        Returns:
            Tuple of (best_estimator, best_params).
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Data not preprocessed. Call preprocess_data() first.")

        logger.info(f"Tuning hyperparameters for {algorithm}...")

        base_model = self._create_model(algorithm)
        param_grid = PARAM_GRIDS[algorithm]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        grid_search.fit(self.X_train, self.y_train)

        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV F1 (macro): {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, use_test: bool = False) -> dict:
        """Evaluate the trained model comprehensively.

        Computes: accuracy, precision, recall, F1 (macro & weighted),
        ROC-AUC (OvR), 5-fold CV accuracy, confusion matrix,
        per-class classification report, feature importances.

        Args:
            use_test: If True, evaluate on test set; else validation set.

        Returns:
            Dictionary of all evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        X_eval = self.X_test if use_test else self.X_val
        y_eval = self.y_test if use_test else self.y_val
        split_name = "test" if use_test else "validation"

        if X_eval is None or y_eval is None:
            raise RuntimeError("Evaluation data not available.")

        logger.info(f"Evaluating model on {split_name} set...")

        y_pred = self.model.predict(X_eval)

        # Core metrics
        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_eval, y_pred)),
            "precision_macro": float(precision_score(y_eval, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_eval, y_pred, average="macro")),
            "f1_macro": float(f1_score(y_eval, y_pred, average="macro")),
            "precision_weighted": float(precision_score(y_eval, y_pred, average="weighted")),
            "recall_weighted": float(recall_score(y_eval, y_pred, average="weighted")),
            "f1_weighted": float(f1_score(y_eval, y_pred, average="weighted")),
        }

        # ROC-AUC (one-vs-rest)
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_eval)
            classes = sorted(np.unique(y_eval))
            if len(classes) > 2:
                y_bin = label_binarize(y_eval, classes=classes)
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
                )
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
                )

        # Cross-validation on training data
        if self.X_train is not None and self.y_train is not None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                self.model, self.X_train, self.y_train,
                cv=cv, scoring="accuracy",
            )
            metrics["cv_accuracy_mean"] = float(cv_scores.mean())
            metrics["cv_accuracy_std"] = float(cv_scores.std())

        # Log summary
        logger.info(f"Evaluation Results ({split_name}):")
        for name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {name}: {value:.4f}")

        # Classification report
        report = classification_report(
            y_eval, y_pred, target_names=list(FERTILITY_LABELS.values()),
        )
        logger.info(f"\nClassification Report:\n{report}")

        cm = confusion_matrix(y_eval, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        metrics["confusion_matrix"] = cm.tolist()

        # Feature importance (tree-based models)
        if hasattr(self.model, "feature_importances_"):
            importances = dict(
                zip(self.feature_names, self.model.feature_importances_.tolist())
            )
            logger.info("\nFeature Importances:")
            for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
                logger.info(f"  {name}: {imp:.4f}")
            metrics["feature_importances"] = importances

        # Per-class report dict
        report_dict = classification_report(
            y_eval, y_pred,
            target_names=list(FERTILITY_LABELS.values()),
            output_dict=True,
        )
        metrics["classification_report"] = report_dict

        return metrics

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------

    def save_model(self, filename: str = "random_forest_model.joblib") -> Path:
        """Save the trained model, optional scaler, and feature names.

        Args:
            filename: Model filename.

        Returns:
            Path to saved model file.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.output_dir / filename
        logger.info(f"Saving model to: {model_path}")
        joblib.dump(self.model, model_path)

        if self.scaler is not None:
            scaler_path = self.output_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to: {scaler_path}")

        features_path = self.output_dir / "feature_names.json"
        with open(features_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Saved feature names to: {features_path}")

        return model_path

    def _save_evaluation_artifacts(self, metrics: dict) -> list[str]:
        """Save evaluation data and plots as artifacts for MLflow."""
        artifact_paths: list[str] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # JSON artifacts
        if "confusion_matrix" in metrics:
            p = self.output_dir / "confusion_matrix.json"
            with open(p, "w") as f:
                json.dump(
                    {"matrix": metrics["confusion_matrix"],
                     "labels": list(FERTILITY_LABELS.values())},
                    f, indent=2,
                )
            artifact_paths.append(str(p))

        if "classification_report" in metrics:
            p = self.output_dir / "classification_report.json"
            with open(p, "w") as f:
                json.dump(metrics["classification_report"], f, indent=2)
            artifact_paths.append(str(p))

        if "feature_importances" in metrics:
            p = self.output_dir / "feature_importances.json"
            with open(p, "w") as f:
                json.dump(metrics["feature_importances"], f, indent=2)
            artifact_paths.append(str(p))

        # Plot artifacts
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Confusion matrix heatmap
            if "confusion_matrix" in metrics:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    np.array(metrics["confusion_matrix"]),
                    annot=True, fmt="d", cmap="Blues",
                    xticklabels=list(FERTILITY_LABELS.values()),
                    yticklabels=list(FERTILITY_LABELS.values()),
                    ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                p = self.output_dir / "confusion_matrix.png"
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                artifact_paths.append(str(p))

            # Feature importance bar chart
            if "feature_importances" in metrics:
                importances = metrics["feature_importances"]
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                names, values = zip(*sorted_features)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(names)), values, color="steelblue")
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names)
                ax.invert_yaxis()
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importances")
                p = self.output_dir / "feature_importances.png"
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                artifact_paths.append(str(p))

            # ROC curves
            if hasattr(self.model, "predict_proba") and self.X_val is not None and self.y_val is not None:
                from sklearn.metrics import auc as sklearn_auc
                from sklearn.metrics import roc_curve

                y_proba = self.model.predict_proba(self.X_val)
                classes = sorted(np.unique(self.y_val))
                if len(classes) > 2:
                    y_bin = np.asarray(label_binarize(self.y_val, classes=classes))
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for i, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                        roc_auc = sklearn_auc(fpr, tpr)
                        lbl = FERTILITY_LABELS.get(cls, str(cls))
                        ax.plot(fpr, tpr, label=f"{lbl} (AUC={roc_auc:.3f})")
                    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("ROC Curves (One-vs-Rest)")
                    ax.legend()
                    p = self.output_dir / "roc_curves.png"
                    fig.savefig(p, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    artifact_paths.append(str(p))

        except ImportError:
            logger.warning("matplotlib/seaborn not installed – skipping plots")

        return artifact_paths

    # ------------------------------------------------------------------
    # MLflow run
    # ------------------------------------------------------------------

    def run_with_mlflow(
        self,
        run_name: str = "fertility-training",
        register_model: bool = True,
        algorithm: str = "random_forest",
        tune_hyperparams: bool = False,
        use_feature_engineering: bool = True,
        **hyperparams: Any,
    ) -> str:
        """Run the full training pipeline with MLflow experiment tracking.

        Args:
            run_name: MLflow run name.
            register_model: Whether to register the model.
            algorithm: Classifier algorithm name.
            tune_hyperparams: Whether to run GridSearchCV.
            use_feature_engineering: Whether to add engineered features.
            **hyperparams: Additional hyperparameters.

        Returns:
            MLflow run ID.
        """
        init_mlflow()

        params: Dict[str, Any] = {
            "test_size": 0.15,
            "val_size": 0.15,
            "apply_log_transform": True,
            "use_scaling": algorithm in ("svm", "knn"),
        }
        params.update(hyperparams)

        run_id: str = ""
        with MLflowRunContext(run_name, self.experiment_name) as run:
            mlflow.log_params({
                "algorithm": algorithm,
                "tune_hyperparams": tune_hyperparams,
                "use_feature_engineering": use_feature_engineering,
                "test_size": params["test_size"],
                "val_size": params["val_size"],
                "apply_log_transform": params["apply_log_transform"],
                "use_scaling": params["use_scaling"],
                "random_state": self.random_state,
            })

            df = self.load_data()
            self.preprocess_data(
                df,
                test_size=params["test_size"],
                val_size=params["val_size"],
                apply_log_transform=params["apply_log_transform"],
                use_feature_engineering=use_feature_engineering,
                use_scaling=params["use_scaling"],
            )

            if self.X_train is None:
                raise RuntimeError("Data preprocessing failed")

            mlflow.log_params({
                "train_samples": len(self.X_train),
                "val_samples": len(self.X_val) if self.X_val is not None else 0,
                "test_samples": len(self.X_test) if self.X_test is not None else 0,
                "n_features": self.X_train.shape[1],
            })

            # Train
            self.train(
                algorithm=algorithm,
                tune_hyperparams=tune_hyperparams,
                **hyperparams,
            )

            if self.best_params:
                safe_params = {
                    f"best_{k}": str(v) for k, v in self.best_params.items()
                }
                mlflow.log_params(safe_params)

            # Evaluate on validation
            val_metrics = self.evaluate(use_test=False)
            for name, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"val_{name}", value)

            # Evaluate on test
            test_metrics = self.evaluate(use_test=True)
            for name, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"test_{name}", value)

            # Artifacts
            artifact_paths = self._save_evaluation_artifacts(val_metrics)
            for path in artifact_paths:
                mlflow.log_artifact(path)

            # Log model
            registered_name = "fertility-predictor" if register_model else None
            mlflow.sklearn.log_model(  # type: ignore[attr-defined]
                self.model,
                artifact_path="model",
                registered_model_name=registered_name,
            )

            self.save_model()
            run_id = run.info.run_id

        return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train soil fertility prediction model (enhanced)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data", type=Path,
        default=PROJECT_ROOT / "data" / "dataset1.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "artifacts" / "fertility_predictor",
        help="Output directory for trained model",
    )
    parser.add_argument("--experiment", type=str, default="fertility-model")
    parser.add_argument("--run-name", type=str, default="fertility-training")
    parser.add_argument(
        "--algorithm", type=str, default="random_forest",
        choices=FertilityTrainer.SUPPORTED_ALGORITHMS,
        help="Classifier algorithm to use",
    )
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--no-feature-engineering", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Soil Fertility Model Training")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info("=" * 60)

    trainer = FertilityTrainer(
        data_path=args.data,
        output_dir=args.output,
        experiment_name=args.experiment,
        random_state=args.seed,
    )

    use_fe = not args.no_feature_engineering

    if args.no_mlflow:
        df = trainer.load_data()
        trainer.preprocess_data(
            df,
            test_size=args.test_size,
            use_feature_engineering=use_fe,
            use_scaling=args.algorithm in ("svm", "knn"),
        )
        trainer.train(
            algorithm=args.algorithm,
            tune_hyperparams=args.tune,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        metrics = trainer.evaluate(use_test=False)
        trainer._save_evaluation_artifacts(metrics)
        trainer.evaluate(use_test=True)
        trainer.save_model()
        logger.info("Training complete (MLflow disabled)")
    else:
        run_id = trainer.run_with_mlflow(
            run_name=args.run_name,
            register_model=not args.no_register,
            algorithm=args.algorithm,
            tune_hyperparams=args.tune,
            use_feature_engineering=use_fe,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            test_size=args.test_size,
        )
        logger.info(f"Training complete. MLflow run ID: {run_id}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

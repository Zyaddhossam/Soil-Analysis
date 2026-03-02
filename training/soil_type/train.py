"""Soil type classification model training script with transfer learning.

Supports multiple CNN backbones (EfficientNet-B0, MobileNetV2, Xception)
with fine-tuning, comprehensive evaluation, and MLflow tracking.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import mlflow
import mlflow.keras  # type: ignore[import-untyped]
import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]
from tensorflow import keras  # type: ignore[import-untyped]
from tensorflow.keras import layers  # type: ignore[import-untyped]
from tensorflow.keras.applications import (  # type: ignore[import-untyped]
    EfficientNetB0,
    MobileNetV2,
    Xception,
)
from tensorflow.keras.callbacks import (  # type: ignore[import-untyped]
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore[import-untyped]

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.constants import NUM_SOIL_TYPES, SOIL_TYPE_LABELS
from src.core.logging import get_logger, setup_logging
from src.utils.mlflow_utils import MLflowRunContext, init_mlflow

# Initialize logging
setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backbone configurations
# ---------------------------------------------------------------------------

BackboneName = Literal["efficientnet_b0", "mobilenet_v2", "xception"]

BACKBONE_CONFIG: dict[str, dict[str, Any]] = {
    "efficientnet_b0": {
        "class": EfficientNetB0,
        "input_size": (224, 224),
        "fine_tune_at": 200,   # unfreeze from this layer index
        "preprocess": "tf",
    },
    "mobilenet_v2": {
        "class": MobileNetV2,
        "input_size": (224, 224),
        "fine_tune_at": 100,
        "preprocess": "tf",
    },
    "xception": {
        "class": Xception,
        "input_size": (299, 299),
        "fine_tune_at": 100,
        "preprocess": "tf",
    },
}


class SoilTypeTrainer:
    """Trainer for soil type classification with multiple backbone support.

    ## Backbones
    - **EfficientNet-B0** (primary): best accuracy/size trade-off.
    - **MobileNetV2** (benchmark): lightweight, mobile-friendly.
    - **Xception** (legacy): original backbone used in v1.

    All backbones are pre-trained on ImageNet, with a custom classification
    head for 4 soil types (Alluvial, Black, Clay, Red).
    """

    SUPPORTED_BACKBONES: list[str] = list(BACKBONE_CONFIG.keys())

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        experiment_name: str = "soil-type-model",
        backbone: BackboneName = "efficientnet_b0",
        random_state: int = 42,
    ):
        """Initialize trainer.

        Args:
            data_dir: Directory containing class subdirectories with images.
            output_dir: Directory to save trained model.
            experiment_name: MLflow experiment name.
            backbone: CNN backbone to use.
            random_state: Random seed for reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.backbone_name = backbone
        self.random_state = random_state

        cfg = BACKBONE_CONFIG[backbone]
        self.image_size: tuple[int, int] = cfg["input_size"]
        self.backbone_class = cfg["class"]
        self.fine_tune_at: int = cfg["fine_tune_at"]

        self.model: Any = None
        self.base_model: Any = None
        self.train_generator: Any = None
        self.val_generator: Any = None
        self.history: Any = None

        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Data generators
    # ------------------------------------------------------------------

    def setup_data_generators(
        self,
        batch_size: int = 32,
        validation_split: float = 0.2,
        augmentation: bool = True,
    ) -> Tuple[Any, Any]:
        """Setup training and validation data generators.

        Uses enhanced augmentation with rotation, shift, zoom,
        brightness, and horizontal-flip for robustness.

        Args:
            batch_size: Batch size for training.
            validation_split: Fraction for validation.
            augmentation: Whether to apply data augmentation.

        Returns:
            Tuple of (train_generator, val_generator).
        """
        logger.info(f"Setting up data generators from: {self.data_dir}")
        logger.info(f"Image size: {self.image_size}")

        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                zoom_range=0.25,
                horizontal_flip=True,
                brightness_range=[0.7, 1.3],
                channel_shift_range=20,
                fill_mode="nearest",
                validation_split=validation_split,
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                validation_split=validation_split,
            )

        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=validation_split,
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            seed=self.random_state,
        )

        self.val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            seed=self.random_state,
        )

        logger.info(f"Class indices: {self.train_generator.class_indices}")
        logger.info(f"Training samples: {self.train_generator.samples}")
        logger.info(f"Validation samples: {self.val_generator.samples}")

        return self.train_generator, self.val_generator

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def build_model(
        self,
        dropout_rate: float = 0.4,
        dense_units: int = 128,
        learning_rate: float = 0.001,
        freeze_base: bool = True,
        label_smoothing: float = 0.1,
    ) -> keras.Model:
        """Build the model with the selected backbone.

        Architecture: Backbone → GlobalAvgPool → Dropout → Dense → Dropout → Softmax

        Args:
            dropout_rate: Dropout rate for regularization.
            dense_units: Units in the dense classification head.
            learning_rate: Initial learning rate.
            freeze_base: Whether to freeze base model weights initially.
            label_smoothing: Label smoothing factor for loss.

        Returns:
            Compiled Keras model.
        """
        logger.info(f"Building model with backbone: {self.backbone_name}")

        self.base_model = self.backbone_class(
            weights="imagenet",
            include_top=False,
            input_shape=(*self.image_size, 3),
        )

        if freeze_base:
            self.base_model.trainable = False
            logger.info("Base model frozen")
        else:
            self.base_model.trainable = True
            logger.info("Base model trainable")

        inputs = keras.Input(shape=(*self.image_size, 3))
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(NUM_SOIL_TYPES, activation="softmax")(x)

        self.model = keras.Model(inputs, outputs, name=f"soil_{self.backbone_name}")

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing,
            ),
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        total_params = self.model.count_params()
        trainable_params = sum(
            tf.size(w).numpy() for w in self.model.trainable_weights
        )
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")

        return self.model

    def get_callbacks(
        self,
        patience_early: int = 8,
        patience_lr: int = 3,
    ) -> list:
        """Get training callbacks.

        Includes: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard.

        Args:
            patience_early: Patience for early stopping.
            patience_lr: Patience for learning rate reduction.

        Returns:
            List of Keras callbacks.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience_early,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=patience_lr,
                min_lr=1e-7,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(self.output_dir / "best_model.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            TensorBoard(
                log_dir=str(self.output_dir / "logs"),
                histogram_freq=1,
            ),
        ]

        return callbacks

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int = 20,
        callbacks: Optional[list] = None,
    ) -> dict:
        """Train the model (feature extraction phase).

        Args:
            epochs: Number of training epochs.
            callbacks: Training callbacks (auto-generated if None).

        Returns:
            Training history dict.
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        if self.train_generator is None:
            raise RuntimeError("Data not setup. Call setup_data_generators() first.")

        logger.info(f"Starting training for {epochs} epochs...")

        if callbacks is None:
            callbacks = self.get_callbacks()

        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Feature extraction training complete")
        return self.history.history

    def fine_tune(
        self,
        unfreeze_from: Optional[int] = None,
        epochs: int = 10,
        learning_rate: float = 1e-5,
    ) -> dict:
        """Fine-tune the top layers of the base model.

        Unfreezes layers from `unfreeze_from` onwards and retrains with
        a lower learning rate.

        Args:
            unfreeze_from: Layer index to unfreeze from (default: per backbone).
            epochs: Number of fine-tuning epochs.
            learning_rate: Learning rate for fine-tuning.

        Returns:
            Fine-tuning history dict.
        """
        if self.model is None or self.base_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        unfreeze_from = unfreeze_from or self.fine_tune_at
        logger.info(
            f"Fine-tuning {self.backbone_name}: unfreezing from layer {unfreeze_from}"
        )

        self.base_model.trainable = True
        for layer in self.base_model.layers[:unfreeze_from]:
            layer.trainable = False

        trainable_count = sum(1 for l in self.model.layers if l.trainable)
        logger.info(f"Trainable layers after unfreeze: {trainable_count}")

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        callbacks = self.get_callbacks(patience_early=5, patience_lr=2)

        ft_history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1,
        )

        return ft_history.history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Evaluate model on validation set with comprehensive metrics.

        Computes: accuracy, precision, recall, F1, per-class metrics,
        confusion matrix, and optionally ROC-AUC.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        logger.info("Evaluating model...")

        # Basic TF evaluation
        results = self.model.evaluate(self.val_generator, verbose=0)
        metric_names = self.model.metrics_names
        metrics: dict[str, Any] = dict(zip(metric_names, [float(v) for v in results]))

        # Get all predictions
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        self.val_generator.reset()
        steps = len(self.val_generator)
        for _ in range(steps):
            X_batch, y_batch = next(self.val_generator)
            preds = self.model.predict(X_batch, verbose=0)
            all_y_true.append(np.argmax(y_batch, axis=1))
            all_y_pred.append(np.argmax(preds, axis=1))
            all_y_prob.append(preds)

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        y_prob = np.concatenate(all_y_prob)

        # Scikit-learn metrics
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.preprocessing import label_binarize

        metrics["sklearn_accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro"))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro"))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))

        # ROC-AUC
        classes = sorted(np.unique(y_true))
        if len(classes) > 2:
            y_bin = label_binarize(y_true, classes=classes)
            try:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_bin, y_prob[:, classes], multi_class="ovr", average="macro")
                )
            except ValueError:
                pass

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Per-class report
        class_labels = [
            SOIL_TYPE_LABELS.get(i, str(i)) for i in range(NUM_SOIL_TYPES)
        ]
        report = classification_report(
            y_true, y_pred, target_names=class_labels, output_dict=True,
        )
        metrics["classification_report"] = report

        # History summary
        if self.history is not None:
            metrics["best_val_accuracy"] = float(max(self.history.history["val_accuracy"]))
            metrics["total_epochs"] = len(self.history.history["loss"])

        logger.info("Evaluation Results:")
        for name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {name}: {value:.4f}")

        report_str = classification_report(y_true, y_pred, target_names=class_labels)
        logger.info(f"\nClassification Report:\n{report_str}")
        logger.info(f"\nConfusion Matrix:\n{cm}")

        return metrics

    # ------------------------------------------------------------------
    # Artifact saving
    # ------------------------------------------------------------------

    def save_model(self, filename: str = "best_model.h5") -> Path:
        """Save trained model and metadata.

        Args:
            filename: Output filename.

        Returns:
            Path to saved model.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.output_dir / filename

        logger.info(f"Saving model to: {model_path}")
        self.model.save(model_path)

        # Class mapping
        class_indices = self.train_generator.class_indices if self.train_generator else {}
        mapping_path = self.output_dir / "class_names.json"
        with open(mapping_path, "w") as f:
            json.dump(class_indices, f, indent=2)

        # Model metadata
        metadata = {
            "backbone": self.backbone_name,
            "image_size": list(self.image_size),
            "num_classes": NUM_SOIL_TYPES,
            "class_labels": {str(k): v for k, v in SOIL_TYPE_LABELS.items()},
        }
        meta_path = self.output_dir / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved class mapping and metadata")
        return model_path

    def _save_evaluation_artifacts(self, metrics: dict) -> list[str]:
        """Save evaluation plots and JSON artifacts.

        Returns:
            List of saved artifact file paths.
        """
        artifact_paths: list[str] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # JSON artifacts
        if "confusion_matrix" in metrics:
            p = self.output_dir / "confusion_matrix.json"
            labels = [SOIL_TYPE_LABELS.get(i, str(i)) for i in range(NUM_SOIL_TYPES)]
            with open(p, "w") as f:
                json.dump({"matrix": metrics["confusion_matrix"], "labels": labels}, f, indent=2)
            artifact_paths.append(str(p))

        if "classification_report" in metrics:
            p = self.output_dir / "classification_report.json"
            with open(p, "w") as f:
                json.dump(metrics["classification_report"], f, indent=2)
            artifact_paths.append(str(p))

        # Plots
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            class_labels = [SOIL_TYPE_LABELS.get(i, str(i)) for i in range(NUM_SOIL_TYPES)]

            # Confusion matrix heatmap
            if "confusion_matrix" in metrics:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    np.array(metrics["confusion_matrix"]),
                    annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_labels,
                    yticklabels=class_labels,
                    ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix ({self.backbone_name})")
                p = self.output_dir / "confusion_matrix.png"
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                artifact_paths.append(str(p))

            # Training history curves
            if self.history is not None:
                hist = self.history.history

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Accuracy
                axes[0].plot(hist["accuracy"], label="Train")
                axes[0].plot(hist["val_accuracy"], label="Validation")
                axes[0].set_title("Model Accuracy")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Accuracy")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # Loss
                axes[1].plot(hist["loss"], label="Train")
                axes[1].plot(hist["val_loss"], label="Validation")
                axes[1].set_title("Model Loss")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Loss")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                fig.suptitle(f"Training History ({self.backbone_name})")
                p = self.output_dir / "training_history.png"
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                artifact_paths.append(str(p))

            # Per-class F1 bar chart
            if "classification_report" in metrics:
                report = metrics["classification_report"]
                names = []
                f1_scores = []
                for lbl in class_labels:
                    if lbl in report:
                        names.append(lbl)
                        f1_scores.append(report[lbl]["f1-score"])

                if names:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(names, f1_scores, color="steelblue")
                    ax.set_ylabel("F1-Score")
                    ax.set_title(f"Per-Class F1 ({self.backbone_name})")
                    ax.set_ylim(0, 1)
                    for i, v in enumerate(f1_scores):
                        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
                    p = self.output_dir / "per_class_f1.png"
                    fig.savefig(p, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    artifact_paths.append(str(p))

        except ImportError:
            logger.warning("matplotlib/seaborn not installed – skipping plots")

        return artifact_paths

    # ------------------------------------------------------------------
    # MLflow integration
    # ------------------------------------------------------------------

    def run_with_mlflow(
        self,
        run_name: str = "soil-type-training",
        register_model: bool = True,
        fine_tune: bool = True,
        **hyperparams: Any,
    ) -> str:
        """Run full training pipeline with MLflow experiment tracking.

        Args:
            run_name: MLflow run name.
            register_model: Whether to register the model.
            fine_tune: Whether to fine-tune after feature extraction.
            **hyperparams: Override default hyperparameters.

        Returns:
            MLflow run ID.
        """
        init_mlflow()

        params: dict[str, Any] = {
            "batch_size": 32,
            "epochs": 20,
            "dropout_rate": 0.4,
            "dense_units": 128,
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "augmentation": True,
            "freeze_base": True,
            "label_smoothing": 0.1,
            "fine_tune_epochs": 10,
            "fine_tune_lr": 1e-5,
            "unfreeze_from": None,
        }
        params.update(hyperparams)

        run_id: str = ""
        with MLflowRunContext(run_name, self.experiment_name) as run:
            mlflow.log_params({
                "backbone": self.backbone_name,
                "image_size": str(self.image_size),
                "batch_size": params["batch_size"],
                "epochs": params["epochs"],
                "dropout_rate": params["dropout_rate"],
                "dense_units": params["dense_units"],
                "learning_rate": params["learning_rate"],
                "validation_split": params["validation_split"],
                "augmentation": params["augmentation"],
                "freeze_base": params["freeze_base"],
                "label_smoothing": params["label_smoothing"],
                "fine_tune": fine_tune,
            })

            # Data
            self.setup_data_generators(
                batch_size=params["batch_size"],
                validation_split=params["validation_split"],
                augmentation=params["augmentation"],
            )

            if self.train_generator is None or self.val_generator is None:
                raise RuntimeError("Data generators not initialized")

            mlflow.log_param("train_samples", self.train_generator.samples)
            mlflow.log_param("val_samples", self.val_generator.samples)

            # Build
            self.build_model(
                dropout_rate=params["dropout_rate"],
                dense_units=params["dense_units"],
                learning_rate=params["learning_rate"],
                freeze_base=params["freeze_base"],
                label_smoothing=params["label_smoothing"],
            )

            # Feature extraction training
            history = self.train(epochs=params["epochs"])
            self._log_history(history, step_offset=0)

            # Fine-tune
            if fine_tune:
                mlflow.log_params({
                    "fine_tune_epochs": params["fine_tune_epochs"],
                    "fine_tune_lr": params["fine_tune_lr"],
                })
                ft_history = self.fine_tune(
                    unfreeze_from=params["unfreeze_from"],
                    epochs=params["fine_tune_epochs"],
                    learning_rate=params["fine_tune_lr"],
                )
                self._log_history(ft_history, step_offset=len(history["loss"]))

            # Evaluate
            metrics = self.evaluate()
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(name, value)

            # Artifacts
            artifact_paths = self._save_evaluation_artifacts(metrics)
            for path in artifact_paths:
                mlflow.log_artifact(path)

            # Log model
            registered_name = "soil-classifier" if register_model else None
            mlflow.keras.log_model(  # type: ignore[attr-defined]
                self.model,
                artifact_path="model",
                registered_model_name=registered_name,
            )

            self.save_model()
            run_id = run.info.run_id

        return run_id

    @staticmethod
    def _log_history(history: dict, step_offset: int = 0) -> None:
        """Log per-epoch training history to MLflow."""
        keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
        n_epochs = len(history.get("loss", []))
        for epoch in range(n_epochs):
            log_metrics: dict[str, float] = {}
            for k in keys:
                if k in history:
                    prefix = "train_" if not k.startswith("val_") else ""
                    metric_name = f"{prefix}{k}" if prefix else k
                    log_metrics[metric_name] = float(history[k][epoch])
            mlflow.log_metrics(log_metrics, step=step_offset + epoch)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train soil type classification model (transfer learning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data", type=Path, required=True,
        help="Path to data directory with class subdirectories",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "artifacts" / "soil_classifier",
        help="Output directory",
    )
    parser.add_argument("--experiment", type=str, default="soil-type-model")
    parser.add_argument("--run-name", type=str, default="soil-type-training")
    parser.add_argument(
        "--backbone", type=str, default="efficientnet_b0",
        choices=SoilTypeTrainer.SUPPORTED_BACKBONES,
        help="CNN backbone architecture",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune base model")
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Soil Type Classification Training (Transfer Learning)")
    logger.info(f"Backbone: {args.backbone}")
    logger.info("=" * 60)

    trainer = SoilTypeTrainer(
        data_dir=args.data,
        output_dir=args.output,
        experiment_name=args.experiment,
        backbone=args.backbone,
        random_state=args.seed,
    )

    if args.no_mlflow:
        trainer.setup_data_generators(batch_size=args.batch_size)
        trainer.build_model(
            dropout_rate=args.dropout,
            dense_units=args.dense_units,
            learning_rate=args.learning_rate,
        )
        trainer.train(epochs=args.epochs)

        if args.fine_tune:
            trainer.fine_tune(
                epochs=args.fine_tune_epochs,
                learning_rate=args.fine_tune_lr,
            )

        trainer.evaluate()
        trainer.save_model()
        logger.info("Training complete (MLflow disabled)")
    else:
        run_id = trainer.run_with_mlflow(
            run_name=args.run_name,
            register_model=not args.no_register,
            fine_tune=args.fine_tune,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout,
            dense_units=args.dense_units,
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune_lr=args.fine_tune_lr,
        )
        logger.info(f"Training complete. MLflow run ID: {run_id}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

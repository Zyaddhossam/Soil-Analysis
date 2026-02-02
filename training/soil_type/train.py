"""Soil type classification model training script.

This module provides training functionality for the Xception-based CNN
soil type classifier with MLflow experiment tracking.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import mlflow
import mlflow.keras  # type: ignore[import-untyped]
import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]
from tensorflow import keras  # type: ignore[import-untyped]
from tensorflow.keras import layers  # type: ignore[import-untyped]
from tensorflow.keras.applications import Xception  # type: ignore[import-untyped]
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

from src.core.constants import NUM_SOIL_TYPES
from src.core.logging import setup_logging, get_logger
from src.utils.mlflow_utils import init_mlflow, MLflowRunContext

# Initialize logging
setup_logging()
logger = get_logger(__name__)


class SoilTypeTrainer:
    """Trainer class for soil type classification model."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        experiment_name: str = "soil-type-model",
        image_size: Tuple[int, int] = (299, 299),
        random_state: int = 42,
    ):
        """Initialize trainer.

        Args:
            data_dir: Directory containing train/val subdirectories.
            output_dir: Directory to save trained model.
            experiment_name: MLflow experiment name.
            image_size: Input image dimensions.
            random_state: Random seed for reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.image_size = image_size
        self.random_state = random_state

        self.model: Any = None
        self.train_generator: Any = None
        self.val_generator: Any = None
        self.history: Any = None

        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    def setup_data_generators(
        self,
        batch_size: int = 32,
        validation_split: float = 0.2,
        augmentation: bool = True,
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """Setup training and validation data generators.

        Args:
            batch_size: Batch size for training.
            validation_split: Fraction for validation.
            augmentation: Whether to apply data augmentation.

        Returns:
            Tuple of (train_generator, val_generator).
        """
        logger.info(f"Setting up data generators from: {self.data_dir}")

        # Training data augmentation
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode="nearest",
                validation_split=validation_split,
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                validation_split=validation_split,
            )

        # Validation - no augmentation
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=validation_split,
        )

        # Create generators
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

        # Log class mapping
        logger.info(f"Class indices: {self.train_generator.class_indices}")
        logger.info(f"Training samples: {self.train_generator.samples}")
        logger.info(f"Validation samples: {self.val_generator.samples}")

        return self.train_generator, self.val_generator

    def build_model(
        self,
        dropout_rate: float = 0.4,
        dense_units: int = 128,
        learning_rate: float = 0.001,
        freeze_base: bool = True,
    ) -> keras.Model:
        """Build the Xception-based model.

        Args:
            dropout_rate: Dropout rate for regularization.
            dense_units: Units in dense layer.
            learning_rate: Initial learning rate.
            freeze_base: Whether to freeze base model weights.

        Returns:
            Compiled Keras model.
        """
        logger.info("Building Xception model...")

        # Base model
        base_model = Xception(
            weights="imagenet",
            include_top=False,
            input_shape=(*self.image_size, 3),
        )

        # Freeze base model
        if freeze_base:
            base_model.trainable = False
            logger.info("Base model frozen")
        else:
            logger.info("Base model trainable")

        # Build model
        inputs = keras.Input(shape=(*self.image_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(NUM_SOIL_TYPES, activation="softmax")(x)

        self.model = keras.Model(inputs, outputs)

        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adamax(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info(f"Model built with {self.model.count_params():,} parameters")
        return self.model

    def get_callbacks(
        self,
        patience_early: int = 5,
        patience_lr: int = 3,
    ) -> list:
        """Get training callbacks.

        Args:
            patience_early: Patience for early stopping.
            patience_lr: Patience for learning rate reduction.

        Returns:
            List of callbacks.
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

    def train(
        self,
        epochs: int = 10,
        callbacks: Optional[list] = None,
    ) -> dict:
        """Train the model.

        Args:
            epochs: Number of training epochs.
            callbacks: Training callbacks.

        Returns:
            Training history.
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

        logger.info("Training complete")
        return self.history.history

    def evaluate(self) -> dict:
        """Evaluate model on validation set.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        logger.info("Evaluating model...")

        # Get predictions
        val_loss, val_accuracy = self.model.evaluate(
            self.val_generator,
            verbose=0,
        )

        metrics = {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        # Additional metrics from history
        if self.history is not None:
            metrics["final_train_loss"] = self.history.history["loss"][-1]
            metrics["final_train_accuracy"] = self.history.history["accuracy"][-1]
            metrics["best_val_accuracy"] = max(self.history.history["val_accuracy"])
            metrics["total_epochs"] = len(self.history.history["loss"])

        logger.info("Evaluation Results:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")

        return metrics

    def save_model(self, filename: str = "model.h5") -> Path:
        """Save trained model.

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
        self.model.save(model_path)

        # Also save class mapping
        class_indices = self.train_generator.class_indices if self.train_generator else {}
        mapping_path = self.output_dir / "class_names.json"
        with open(mapping_path, "w") as f:
            json.dump(class_indices, f, indent=2)
        logger.info(f"Saved class mapping to: {mapping_path}")

        return model_path

    def fine_tune(
        self,
        unfreeze_layers: int = 20,
        epochs: int = 5,
        learning_rate: float = 1e-5,
    ) -> dict:
        """Fine-tune the model by unfreezing top layers.

        Args:
            unfreeze_layers: Number of layers to unfreeze.
            epochs: Number of fine-tuning epochs.
            learning_rate: Learning rate for fine-tuning.

        Returns:
            Fine-tuning history.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        logger.info(f"Fine-tuning: unfreezing {unfreeze_layers} layers...")

        # Unfreeze top layers of base model
        base_model = self.model.layers[1]  # Xception is the second layer
        base_model.trainable = True

        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        trainable_count = sum(
            1 for layer in self.model.layers if layer.trainable
        )
        logger.info(f"Trainable layers: {trainable_count}")

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train
        callbacks = self.get_callbacks(patience_early=3, patience_lr=2)

        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1,
        )

        return history.history

    def run_with_mlflow(
        self,
        run_name: str = "soil-type-training",
        register_model: bool = True,
        fine_tune: bool = False,
        **hyperparams,
    ) -> str:
        """Run training with MLflow tracking.

        Args:
            run_name: Name for the MLflow run.
            register_model: Whether to register model.
            fine_tune: Whether to perform fine-tuning.
            **hyperparams: Model hyperparameters.

        Returns:
            MLflow run ID.
        """
        # Initialize MLflow
        init_mlflow()

        # Default hyperparameters
        params = {
            "batch_size": 32,
            "epochs": 10,
            "dropout_rate": 0.4,
            "dense_units": 128,
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "augmentation": True,
            "freeze_base": True,
            "fine_tune_epochs": 5,
            "fine_tune_lr": 1e-5,
            "unfreeze_layers": 20,
        }
        params.update(hyperparams)

        run_id: str = ""
        with MLflowRunContext(run_name, self.experiment_name) as run:
            # Log parameters
            mlflow.log_params({
                "batch_size": params["batch_size"],
                "epochs": params["epochs"],
                "dropout_rate": params["dropout_rate"],
                "dense_units": params["dense_units"],
                "learning_rate": params["learning_rate"],
                "validation_split": params["validation_split"],
                "augmentation": params["augmentation"],
                "freeze_base": params["freeze_base"],
                "image_size": str(self.image_size),
                "fine_tune": fine_tune,
            })

            # Setup data
            self.setup_data_generators(
                batch_size=params["batch_size"],
                validation_split=params["validation_split"],
                augmentation=params["augmentation"],
            )

            if self.train_generator is None or self.val_generator is None:
                raise RuntimeError("Data generators not initialized")

            mlflow.log_param("train_samples", self.train_generator.samples)
            mlflow.log_param("val_samples", self.val_generator.samples)

            # Build model
            self.build_model(
                dropout_rate=params["dropout_rate"],
                dense_units=params["dense_units"],
                learning_rate=params["learning_rate"],
                freeze_base=params["freeze_base"],
            )

            # Train
            history = self.train(epochs=params["epochs"])

            # Log training metrics
            for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history["loss"],
                history["accuracy"],
                history["val_loss"],
                history["val_accuracy"],
            )):
                mlflow.log_metrics({
                    "train_loss": loss,
                    "train_accuracy": acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }, step=epoch)

            # Fine-tune if requested
            if fine_tune:
                mlflow.log_params({
                    "fine_tune_epochs": params["fine_tune_epochs"],
                    "fine_tune_lr": params["fine_tune_lr"],
                    "unfreeze_layers": params["unfreeze_layers"],
                })

                ft_history = self.fine_tune(
                    unfreeze_layers=params["unfreeze_layers"],
                    epochs=params["fine_tune_epochs"],
                    learning_rate=params["fine_tune_lr"],
                )

                base_epoch = len(history["loss"])
                for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                    ft_history["loss"],
                    ft_history["accuracy"],
                    ft_history["val_loss"],
                    ft_history["val_accuracy"],
                )):
                    mlflow.log_metrics({
                        "train_loss": loss,
                        "train_accuracy": acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    }, step=base_epoch + epoch)

            # Evaluate
            metrics = self.evaluate()
            mlflow.log_metrics({
                "final_val_loss": metrics["val_loss"],
                "final_val_accuracy": metrics["val_accuracy"],
            })

            # Log model
            registered_name = "soil-classifier" if register_model else None
            mlflow.keras.log_model(  # type: ignore[attr-defined]
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
        description="Train soil type classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to data directory with class subdirectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "soil_classifier",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="soil-type-model",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="soil-type-training",
        help="MLflow run name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout rate",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Perform fine-tuning after initial training",
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
    logger.info("Soil Type Classification Model Training")
    logger.info("=" * 60)

    trainer = SoilTypeTrainer(
        data_dir=args.data,
        output_dir=args.output,
        experiment_name=args.experiment,
        random_state=args.seed,
    )

    if args.no_mlflow:
        # Train without MLflow
        trainer.setup_data_generators(batch_size=args.batch_size)
        trainer.build_model(
            dropout_rate=args.dropout,
            learning_rate=args.learning_rate,
        )
        trainer.train(epochs=args.epochs)

        if args.fine_tune:
            trainer.fine_tune()

        trainer.evaluate()
        trainer.save_model()
        logger.info("Training complete (MLflow disabled)")
    else:
        # Train with MLflow tracking
        run_id = trainer.run_with_mlflow(
            run_name=args.run_name,
            register_model=not args.no_register,
            fine_tune=args.fine_tune,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout,
        )
        logger.info(f"Training complete. MLflow run ID: {run_id}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

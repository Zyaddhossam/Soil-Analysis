"""MLflow integration utilities."""

import os
from types import TracebackType
from typing import Any, Literal

import mlflow
import mlflow.keras  # type: ignore[import-untyped]
import mlflow.sklearn  # type: ignore[import-untyped]
import mlflow.tensorflow  # type: ignore[import-untyped]
from mlflow.tracking import MlflowClient

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


def init_mlflow() -> MlflowClient:
    """Initialize MLflow with configured tracking URI.

    Returns:
        MlflowClient instance.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    if settings.mlflow_model_registry_uri:
        os.environ["MLFLOW_REGISTRY_URI"] = settings.mlflow_model_registry_uri

    logger.info(f"MLflow tracking URI: {settings.mlflow_tracking_uri}")
    return MlflowClient()


def get_or_create_experiment(name: str | None = None) -> str:
    """Get or create an MLflow experiment.

    Args:
        name: Experiment name. Defaults to configured experiment name.

    Returns:
        Experiment ID.
    """
    name = name or settings.mlflow_experiment_name

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
        logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.debug(f"Using existing experiment '{name}' with ID: {experiment_id}")

    return experiment_id


def load_model_from_registry(
    model_name: str,
    version: str | None = None,
    stage: str | None = None,
) -> Any:
    """Load a model from MLflow model registry.

    Args:
        model_name: Registered model name.
        version: Specific version number. Takes precedence over stage.
        stage: Model stage (e.g., 'Production', 'Staging').

    Returns:
        Loaded model.

    Raises:
        mlflow.exceptions.MlflowException: If model not found.
    """
    if version:
        model_uri = f"models:/{model_name}/{version}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"

    logger.info(f"Loading model from: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def load_sklearn_model_from_registry(
    model_name: str,
    version: str | None = None,
    stage: str | None = None,
) -> Any:
    """Load a scikit-learn model from MLflow model registry.

    Args:
        model_name: Registered model name.
        version: Specific version number.
        stage: Model stage.

    Returns:
        Loaded sklearn model.
    """
    if version:
        model_uri = f"models:/{model_name}/{version}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"

    logger.info(f"Loading sklearn model from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)  # type: ignore[attr-defined]


def load_keras_model_from_registry(
    model_name: str,
    version: str | None = None,
    stage: str | None = None,
) -> Any:
    """Load a Keras model from MLflow model registry.

    Args:
        model_name: Registered model name.
        version: Specific version number.
        stage: Model stage.

    Returns:
        Loaded Keras model.
    """
    if version:
        model_uri = f"models:/{model_name}/{version}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"

    logger.info(f"Loading Keras model from: {model_uri}")
    return mlflow.keras.load_model(model_uri)  # type: ignore[attr-defined]


def log_model_artifact(
    model: Any,
    artifact_path: str,
    registered_model_name: str | None = None,
    flavor: str = "sklearn",
    **kwargs: Any,
) -> str:
    """Log a model as an MLflow artifact.

    Args:
        model: Model to log.
        artifact_path: Path within the artifact store.
        registered_model_name: Name to register the model under.
        flavor: MLflow flavor ('sklearn', 'keras', 'tensorflow').
        **kwargs: Additional arguments for the log function.

    Returns:
        Model URI.
    """
    log_func = {
        "sklearn": mlflow.sklearn.log_model,  # type: ignore[attr-defined]
        "keras": mlflow.keras.log_model,  # type: ignore[attr-defined]
        "tensorflow": mlflow.tensorflow.log_model,  # type: ignore[attr-defined]
    }.get(flavor)

    if log_func is None:
        raise ValueError(f"Unsupported flavor: {flavor}")

    model_info = log_func(
        model,
        artifact_path,
        registered_model_name=registered_model_name,
        **kwargs,
    )

    model_uri: str = model_info.model_uri
    logger.info(f"Logged model to: {model_uri}")
    return model_uri


class MLflowRunContext:
    """Context manager for MLflow runs with automatic cleanup."""

    def __init__(
        self,
        run_name: str,
        experiment_name: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Initialize MLflow run context.

        Args:
            run_name: Name for the run.
            experiment_name: Experiment name.
            tags: Additional tags for the run.
        """
        self.run_name = run_name
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self.tags = tags or {}
        self.run: mlflow.ActiveRun | None = None

    def __enter__(self) -> mlflow.ActiveRun:
        """Start MLflow run."""
        experiment_id = get_or_create_experiment(self.experiment_name)
        mlflow.set_experiment(experiment_id=experiment_id)

        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        logger.info(f"Started MLflow run: {self.run.info.run_id}")
        return self.run

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """End MLflow run."""
        if exc_type is not None:
            mlflow.set_tag("run_status", "failed")
            mlflow.set_tag("error", str(exc_val))
            logger.error(f"MLflow run failed: {exc_val}")
        else:
            mlflow.set_tag("run_status", "success")

        mlflow.end_run()
        if self.run is not None:
            logger.info(f"Ended MLflow run: {self.run.info.run_id}")
        return False  # Don't suppress exceptions

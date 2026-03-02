"""Model comparison and registry management utilities.

Provides functions to compare MLflow runs, promote best models,
and generate comparison reports across experiments.
"""

import json
from pathlib import Path
from typing import Any

import mlflow

from src.core.logging import get_logger
from src.utils.mlflow_utils import init_mlflow

logger = get_logger(__name__)


def compare_runs(
    experiment_name: str,
    metric: str = "val_accuracy",
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Compare MLflow runs by a given metric.

    Args:
        experiment_name: MLflow experiment name.
        metric: Metric to rank by (higher is better).
        top_n: Number of top runs to return.

    Returns:
        List of run summaries sorted by metric (descending).
    """
    client = init_mlflow()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n,
    )

    summaries: list[dict[str, Any]] = []
    for run in runs:
        summary = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "params": dict(run.data.params),
            "metrics": {k: v for k, v in run.data.metrics.items()},
        }
        summaries.append(summary)

    return summaries


def promote_best_model(
    experiment_name: str,
    registered_model_name: str,
    metric: str = "val_accuracy",
    alias: str = "champion",
) -> str | None:
    """Promote the best run's model version to champion alias.

    Args:
        experiment_name: MLflow experiment name.
        registered_model_name: Registry model name.
        metric: Metric to select best run (higher is better).
        alias: Alias to assign (e.g., 'champion').

    Returns:
        Version string of promoted model, or None if no runs found.
    """
    client = init_mlflow()

    runs = compare_runs(experiment_name, metric=metric, top_n=1)
    if not runs:
        logger.warning("No runs found to promote")
        return None

    best_run_id = runs[0]["run_id"]
    logger.info(
        f"Best run: {best_run_id} with {metric}={runs[0]['metrics'].get(metric)}"
    )

    # Find model version for this run
    try:
        versions = client.search_model_versions(
            f"name='{registered_model_name}'"
        )
    except Exception:
        logger.warning(f"No registered model '{registered_model_name}' found")
        return None

    target_version: str | None = None
    for v in versions:
        if v.run_id == best_run_id:
            target_version = v.version
            break

    if target_version is None:
        logger.warning(f"No model version found for run {best_run_id}")
        return None

    # Set alias
    try:
        client.set_registered_model_alias(
            registered_model_name, alias, target_version,
        )
        logger.info(
            f"Promoted {registered_model_name} v{target_version} to '{alias}'"
        )
    except Exception as e:
        logger.warning(f"Could not set alias (MLflow version may not support it): {e}")
        # Fall back to stage transition for older MLflow
        try:
            client.transition_model_version_stage(
                registered_model_name, target_version, "Production",
            )
            logger.info(
                f"Transitioned {registered_model_name} v{target_version} to Production"
            )
        except Exception as e2:
            logger.error(f"Failed to promote model: {e2}")
            return None

    return target_version


def generate_comparison_report(
    experiment_name: str,
    output_path: Path,
    metric: str = "val_accuracy",
    top_n: int = 10,
) -> Path:
    """Generate a JSON comparison report for an experiment.

    Args:
        experiment_name: MLflow experiment name.
        output_path: Path to write the report JSON.
        metric: Primary ranking metric.
        top_n: Number of top runs.

    Returns:
        Path to the generated report.
    """
    summaries = compare_runs(experiment_name, metric=metric, top_n=top_n)

    report = {
        "experiment": experiment_name,
        "ranking_metric": metric,
        "runs": summaries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Comparison report saved to: {output_path}")
    return output_path

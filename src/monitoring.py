"""Model monitoring: drift detection.

Two halves, taken from the teachers' notebooks:

1. After training (01_training.ipynb): run the model on the validation set
   and save a *baseline*, what "normal" data and predictions look like.
2. In production (02_deployment.ipynb): compare the requests the API logged
   in the last days against that baseline with Kolmogorov-Smirnov (KS) tests.
   If the distributions differ (p-value < 0.05), we report drift.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from loguru import logger
import numpy as np
from scipy.stats import ks_2samp
import torch

from src.config import (
    DRIFT_BASELINE_STATS_PATH,
    DRIFT_DEFAULT_DAYS,
    DRIFT_KS_PVALUE_THRESHOLD,
    DRIFT_MEAN_CONFIDENCE_THRESHOLD,
    INFERENCE_LOG_PATH,
    THRESHOLD,
)

# ---------------------------------------------------------------- 1. baseline (after training)


def compute_baseline_statistics(model, data_loader, threshold=THRESHOLD, device="cpu"):
    """Run the model on a dataset (the validation set) and summarise what normal looks like."""
    model.eval()
    model.to(device)

    image_means, image_stds, confidences, predictions = [], [], [], []
    class_counts = {0: 0, 1: 0}

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu().numpy().flatten()

            image_means.append(images.mean(dim=(0, 2, 3)).cpu().numpy())  # per colour channel
            image_stds.append(images.std(dim=(0, 2, 3)).cpu().numpy())
            confidences.extend(float(p) for p in probs)
            predictions.extend(int(p >= threshold) for p in probs)
            for label in labels.tolist():
                class_counts[int(label)] += 1

    image_means, image_stds = np.array(image_means), np.array(image_stds)
    baseline = {
        "image_statistics": {
            f"{stat}_{channel}": float(values[:, i].mean())
            for stat, values in (("mean", image_means), ("std", image_stds))
            for i, channel in enumerate("rgb")
        },
        "class_distribution": {
            "class_0_count": class_counts[0],
            "class_1_count": class_counts[1],
            "total": class_counts[0] + class_counts[1],
        },
        "prediction_distribution": {
            "pred_0_count": predictions.count(0),
            "pred_1_count": predictions.count(1),
            "total": len(predictions),
        },
        "confidence_statistics": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
        },
        # The real validation confidences, so the KS test compares real samples
        "confidences": confidences,
    }
    logger.info(
        f"Baseline: {baseline['class_distribution']['total']} images, "
        f"mean confidence {baseline['confidence_statistics']['mean']:.3f}"
    )
    return baseline


def save_baseline(baseline: dict, path: Path = DRIFT_BASELINE_STATS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2))


def load_baseline(path: Path = DRIFT_BASELINE_STATS_PATH):
    if not path.exists():
        logger.warning(f"No baseline found at {path}")
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------- 2. drift detection (in production)


def load_inference_logs(path: Path = INFERENCE_LOG_PATH) -> list[dict]:
    """The API appends one entry per request: timestamp, prediction, confidence."""
    if not path.exists():
        return []
    try:
        logs = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Inference log {path} is not valid json")
        return []
    return logs if isinstance(logs, list) else []


def filter_logs_by_time(logs: list[dict], days: int = DRIFT_DEFAULT_DAYS, now=None) -> list[dict]:
    """Keep only the requests from the last `days` days."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    recent = []
    for log in logs:
        try:
            if datetime.fromisoformat(log["timestamp"]) >= cutoff:
                recent.append(log)
        except (KeyError, TypeError, ValueError):
            recent.append(log)  # no usable timestamp: keep it rather than lose it
    return recent


def check_confidence_drift(confidences: list[float], baseline: dict) -> dict:
    """KS test: do the model's confidence scores now look like on the validation set?"""
    baseline_confidences = (baseline or {}).get("confidences", [])
    if len(confidences) < 2 or len(baseline_confidences) < 2:
        return {"ks_statistic": 0.0, "ks_pvalue": 1.0, "drift_detected": False}

    ks_statistic, ks_pvalue = ks_2samp(baseline_confidences, confidences)
    mean_diff = abs(float(np.mean(confidences)) - float(np.mean(baseline_confidences)))
    drift = ks_pvalue < DRIFT_KS_PVALUE_THRESHOLD or mean_diff > DRIFT_MEAN_CONFIDENCE_THRESHOLD
    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "mean_diff": mean_diff,
        "drift_detected": bool(drift),
    }


def check_prediction_drift(predictions: list[int], baseline: dict) -> dict:
    """KS test: is the share of clean/defective predictions the same as on the validation set?"""
    counts = (baseline or {}).get("prediction_distribution")
    if not predictions or not counts or counts["total"] == 0:
        return {"ks_statistic": 0.0, "ks_pvalue": 1.0, "drift_detected": False}

    baseline_predictions = [0] * counts["pred_0_count"] + [1] * counts["pred_1_count"]
    ks_statistic, ks_pvalue = ks_2samp(baseline_predictions, predictions)
    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "drift_detected": bool(ks_pvalue < DRIFT_KS_PVALUE_THRESHOLD),
    }


def detect_drift(logs: list[dict], baseline: dict, days: int = DRIFT_DEFAULT_DAYS, now=None) -> dict:
    """Main drift check, what the /drift endpoint returns."""
    recent = filter_logs_by_time(logs, days, now)
    if not recent:
        return {"overall_drift_detected": False, "num_samples": 0, "error": "No logs found"}

    confidences = [float(log["confidence"]) for log in recent if log.get("confidence") is not None]
    predictions = [int(log["prediction"]) for log in recent if log.get("prediction") is not None]

    confidence_drift = check_confidence_drift(confidences, baseline)
    prediction_drift = check_prediction_drift(predictions, baseline)
    return {
        "overall_drift_detected": confidence_drift["drift_detected"] or prediction_drift["drift_detected"],
        "num_samples": len(recent),
        "confidence_drift": confidence_drift,
        "prediction_drift": prediction_drift,
        "prediction_counts": {"class_0": predictions.count(0), "class_1": predictions.count(1)},
    }

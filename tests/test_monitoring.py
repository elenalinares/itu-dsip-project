"""Tests for src/monitoring.py: baseline statistics and drift detection."""

from datetime import datetime, timedelta, timezone

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.monitoring import (
    compute_baseline_statistics,
    detect_drift,
    filter_logs_by_time,
    load_baseline,
    save_baseline,
)

NOW = datetime(2026, 11, 23, tzinfo=timezone.utc)


def log(prediction, confidence, days_ago=0):
    return {
        "timestamp": (NOW - timedelta(days=days_ago)).isoformat(),
        "prediction": prediction,
        "confidence": confidence,
    }


def baseline_with(confidences, pred_0, pred_1):
    return {
        "confidences": confidences,
        "prediction_distribution": {"pred_0_count": pred_0, "pred_1_count": pred_1, "total": pred_0 + pred_1},
    }


# ---------------------------------------------------------------- baseline


class TinyModel(nn.Module):
    """Stand-in for the CNN: 3x64x64 image -> 1 logit."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 64 * 64, 1)

    def forward(self, x):
        return self.fc1(x.flatten(1))


def test_baseline_has_counts_and_confidences():
    torch.manual_seed(0)
    images = torch.randn(10, 3, 64, 64)
    labels = torch.tensor([0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    baseline = compute_baseline_statistics(TinyModel(), loader)

    assert baseline["class_distribution"] == {"class_0_count": 2, "class_1_count": 8, "total": 10}
    assert baseline["prediction_distribution"]["total"] == 10
    assert len(baseline["confidences"]) == 10
    assert all(0 <= c <= 1 for c in baseline["confidences"])
    assert set(baseline["image_statistics"]) == {"mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"}


def test_baseline_save_and_load(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline({"a": 1}, path)
    assert load_baseline(path) == {"a": 1}
    assert load_baseline(tmp_path / "missing.json") is None


# ---------------------------------------------------------------- drift detection


def test_filter_logs_keeps_only_recent_ones():
    logs = [log(1, 0.9, days_ago=1), log(1, 0.9, days_ago=30)]
    assert len(filter_logs_by_time(logs, days=7, now=NOW)) == 1


def test_no_logs_means_no_drift():
    result = detect_drift([], baseline_with([0.9] * 50, 10, 40), now=NOW)
    assert result["overall_drift_detected"] is False
    assert result["num_samples"] == 0


def test_same_data_as_baseline_is_not_drift():
    confidences = [0.1, 0.2, 0.8, 0.85, 0.9, 0.95] * 10
    baseline = baseline_with(confidences, pred_0=20, pred_1=40)
    logs = [log(int(c >= 0.5), c) for c in confidences]

    assert detect_drift(logs, baseline, now=NOW)["overall_drift_detected"] is False


def test_very_different_data_is_drift():
    # baseline: model was confident, mostly "defective"
    baseline = baseline_with([0.9, 0.95, 0.85, 0.92] * 15, pred_0=10, pred_1=50)
    # now: model is unsure and says "clean" all the time
    logs = [log(0, c) for c in [0.3, 0.35, 0.4, 0.45] * 15]

    result = detect_drift(logs, baseline, now=NOW)

    assert result["overall_drift_detected"] is True
    assert result["confidence_drift"]["drift_detected"] is True
    assert result["prediction_drift"]["drift_detected"] is True

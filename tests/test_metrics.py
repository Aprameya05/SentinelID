"""
Unit tests for evaluation metrics (liveness and verification).
These run without any GPU or dataset — pure numpy.
"""

import numpy as np
import pytest

from evaluation.metrics import (
    audit_demographic_fairness,
    compute_liveness_metrics,
    compute_verification_metrics,
)


def _make_scores(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    # Live samples score higher on average
    scores = np.where(labels == 1, rng.normal(0.7, 0.15, n), rng.normal(0.3, 0.15, n))
    return np.clip(scores, 0.0, 1.0), labels


class TestLivenessMetrics:
    def test_output_shape_and_range(self):
        scores, labels = _make_scores()
        m = compute_liveness_metrics(scores, labels)
        assert 0.0 <= m.acer <= 1.0
        assert 0.0 <= m.apcer <= 1.0
        assert 0.0 <= m.bpcer <= 1.0
        assert 0.0 <= m.auc <= 1.0
        assert 0.0 <= m.eer <= 1.0

    def test_acer_equals_apcer_plus_bpcer_over_2(self):
        scores, labels = _make_scores()
        m = compute_liveness_metrics(scores, labels)
        assert abs(m.acer - (m.apcer + m.bpcer) / 2) < 1e-6

    def test_perfect_classifier(self):
        labels = np.array([1] * 100 + [0] * 100)
        scores = np.array([1.0] * 100 + [0.0] * 100)
        m = compute_liveness_metrics(scores, labels)
        assert m.auc > 0.99
        assert m.eer < 0.02

    def test_random_classifier(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, size=500)
        scores = rng.uniform(0, 1, size=500)
        m = compute_liveness_metrics(scores, labels)
        assert 0.35 < m.auc < 0.65  # roughly 0.5

    def test_custom_threshold(self):
        scores, labels = _make_scores()
        m = compute_liveness_metrics(scores, labels, threshold=0.6)
        assert m.threshold == 0.6

    def test_all_spoof(self):
        scores = np.array([0.2, 0.3, 0.1])
        labels = np.array([0, 0, 0])
        # roc_auc_score requires at least 2 classes; should raise or return gracefully
        with pytest.raises(Exception):
            compute_liveness_metrics(scores, labels)

    def test_all_live(self):
        scores = np.array([0.8, 0.9, 0.7])
        labels = np.array([1, 1, 1])
        with pytest.raises(Exception):
            compute_liveness_metrics(scores, labels)


class TestVerificationMetrics:
    def test_output_range(self):
        scores, labels = _make_scores()
        m = compute_verification_metrics(scores, labels)
        assert 0.0 <= m.auc <= 1.0
        assert 0.0 <= m.eer <= 1.0
        assert 0.0 <= m.tar_at_far_1e3 <= 1.0
        assert 0.0 <= m.tar_at_far_1e4 <= 1.0

    def test_tar_ordering(self):
        scores, labels = _make_scores()
        m = compute_verification_metrics(scores, labels)
        # TAR@FAR=0.01% should be <= TAR@FAR=0.1% (stricter threshold)
        assert m.tar_at_far_1e4 <= m.tar_at_far_1e3 + 0.05  # allow small numerical slack

    def test_perfect_verification(self):
        labels = np.array([1] * 50 + [0] * 50)
        scores = np.array([0.95] * 50 + [0.05] * 50)
        m = compute_verification_metrics(scores, labels)
        assert m.auc > 0.99


class TestDemographicFairness:
    def test_basic_report(self):
        rng = np.random.default_rng(1)
        n = 300
        similarities = rng.uniform(0, 1, n)
        labels = rng.integers(0, 2, n)
        groups = rng.choice(["group_a", "group_b", "group_c"], size=n)
        reports = audit_demographic_fairness(similarities, labels, groups, threshold=0.5)
        assert len(reports) == 3
        for r in reports:
            assert 0.0 <= r.fmr <= 1.0
            assert 0.0 <= r.fnmr <= 1.0
            assert r.n_samples > 0

    def test_fmr_fnmr_perfect(self):
        # Perfect classifier: FMR=0, FNMR=0 for all groups
        labels = np.array([1] * 100 + [0] * 100)
        scores = np.array([1.0] * 100 + [0.0] * 100)
        groups = np.array(["a"] * 100 + ["b"] * 100)
        reports = audit_demographic_fairness(scores, labels, groups, threshold=0.5)
        for r in reports:
            assert r.fmr < 0.01
            assert r.fnmr < 0.01

    def test_sample_counts_sum_to_total(self):
        rng = np.random.default_rng(2)
        n = 400
        labels = rng.integers(0, 2, n)
        scores = rng.uniform(0, 1, n)
        groups = rng.choice(["x", "y"], size=n)
        reports = audit_demographic_fairness(scores, labels, groups)
        assert sum(r.n_samples for r in reports) == n

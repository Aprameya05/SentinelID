"""
Evaluation metrics for SentinelID modules.

ACER, APCER, BPCER for liveness.
AUC, EER, TAR@FAR for face recognition and deepfake detection.
Per-demographic FMR/FNMR for bias auditing.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class LivenessMetrics:
    acer: float   # Average Classification Error Rate = (APCER + BPCER) / 2
    apcer: float  # Attack Presentation Classification Error Rate (false accept)
    bpcer: float  # Bona fide Presentation Classification Error Rate (false reject)
    auc: float
    eer: float
    threshold: float


@dataclass
class VerificationMetrics:
    auc: float
    eer: float
    tar_at_far_1e3: float   # TAR @ FAR = 0.1%
    tar_at_far_1e4: float   # TAR @ FAR = 0.01%


@dataclass
class DemographicFairnessReport:
    group_name: str
    fmr: float   # False Match Rate
    fnmr: float  # False Non-Match Rate
    n_samples: int


def compute_liveness_metrics(
    scores: np.ndarray,
    labels: np.ndarray,   # 1 = live, 0 = spoof
    threshold: float | None = None,
) -> LivenessMetrics:
    """
    Compute ISO 30107-3 biometric presentation attack detection metrics.

    APCER: fraction of spoof samples classified as live
    BPCER: fraction of live samples classified as spoof
    ACER: (APCER + BPCER) / 2
    """
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # EER: threshold where FPR == FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])

    if threshold is None:
        threshold = eer_threshold

    preds = (scores >= threshold).astype(int)
    spoof_idx = labels == 0
    live_idx = labels == 1

    apcer = float(preds[spoof_idx].mean()) if spoof_idx.sum() > 0 else 0.0
    bpcer = float((1 - preds[live_idx]).mean()) if live_idx.sum() > 0 else 0.0
    acer = (apcer + bpcer) / 2

    return LivenessMetrics(
        acer=acer, apcer=apcer, bpcer=bpcer,
        auc=float(auc), eer=eer, threshold=threshold,
    )


def compute_verification_metrics(
    similarities: np.ndarray,
    labels: np.ndarray,  # 1 = genuine pair, 0 = impostor pair
) -> VerificationMetrics:
    auc = float(roc_auc_score(labels, similarities))
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    def tar_at_far(target_far: float) -> float:
        idx = np.searchsorted(fpr, target_far)
        idx = min(idx, len(tpr) - 1)
        return float(tpr[idx])

    return VerificationMetrics(
        auc=auc,
        eer=eer,
        tar_at_far_1e3=tar_at_far(1e-3),
        tar_at_far_1e4=tar_at_far(1e-4),
    )


def audit_demographic_fairness(
    similarities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,     # demographic group label per sample pair
    threshold: float = 0.45,
) -> list[DemographicFairnessReport]:
    """
    Per-demographic FMR/FNMR audit.

    A fair system should have similar FMR and FNMR across all demographic groups.
    Large disparities indicate bias in the training data or model.
    """
    reports = []
    for group in np.unique(groups):
        mask = groups == group
        g_sims = similarities[mask]
        g_labels = labels[mask]
        preds = (g_sims >= threshold).astype(int)

        imp_mask = g_labels == 0
        gen_mask = g_labels == 1

        fmr = float(preds[imp_mask].mean()) if imp_mask.sum() > 0 else 0.0
        fnmr = float((1 - preds[gen_mask]).mean()) if gen_mask.sum() > 0 else 0.0

        reports.append(DemographicFairnessReport(
            group_name=str(group),
            fmr=fmr,
            fnmr=fnmr,
            n_samples=int(mask.sum()),
        ))

    return reports


def print_fairness_report(reports: list[DemographicFairnessReport]):
    from rich.console import Console
    from rich.table import Table
    console = Console()

    table = Table(title="Demographic Fairness Audit")
    table.add_column("Group", style="cyan")
    table.add_column("FMR", justify="right", style="red")
    table.add_column("FNMR", justify="right", style="yellow")
    table.add_column("N Samples", justify="right")

    for r in reports:
        table.add_row(r.group_name, f"{r.fmr:.4f}", f"{r.fnmr:.4f}", str(r.n_samples))

    fmrs = [r.fmr for r in reports]
    fnmrs = [r.fnmr for r in reports]
    table.add_row(
        "[bold]Range[/bold]",
        f"[bold]{max(fmrs)-min(fmrs):.4f}[/bold]",
        f"[bold]{max(fnmrs)-min(fnmrs):.4f}[/bold]",
        "",
    )
    console.print(table)

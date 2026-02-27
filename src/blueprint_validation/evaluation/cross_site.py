"""Cross-site discrimination logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..common import get_logger

logger = get_logger("evaluation.cross_site")


@dataclass
class CrossSiteMetrics:
    """Metrics from the cross-site discrimination test."""

    # Classification accuracy per model
    model_accuracies: Dict[str, float]  # {facility_id: accuracy}
    # Confusion matrix: {source_model: {predicted_facility: count}}
    confusion_matrix: Dict[str, Dict[str, int]]
    # Overall discrimination rate
    overall_accuracy: float
    # Per-pair LPIPS distances
    inter_facility_lpips: float
    intra_facility_lpips: float

    def to_dict(self) -> dict:
        return {
            "model_accuracies": self.model_accuracies,
            "confusion_matrix": self.confusion_matrix,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "inter_facility_lpips": round(self.inter_facility_lpips, 4),
            "intra_facility_lpips": round(self.intra_facility_lpips, 4),
        }


def compute_cross_site_metrics(
    classifications: List[Dict],
    lpips_inter: List[float],
    lpips_intra: List[float],
    facility_ids: List[str],
) -> CrossSiteMetrics:
    """Compute cross-site discrimination metrics from VLM classifications.

    Args:
        classifications: List of dicts with keys:
            - source_model: which facility's model generated the clip
            - predicted_facility: which facility the VLM classified it as
            - confidence: VLM confidence score
        lpips_inter: LPIPS distances between cross-facility pairs
        lpips_intra: LPIPS distances between same-facility pairs
        facility_ids: List of facility IDs
    """
    import numpy as np

    # Build confusion matrix
    confusion: Dict[str, Dict[str, int]] = {}
    for fid in facility_ids:
        confusion[fid] = {fid2: 0 for fid2 in facility_ids}

    correct = 0
    total = 0

    for clf in classifications:
        source = clf["source_model"]
        predicted = clf["predicted_facility"]
        if source in confusion and predicted in confusion[source]:
            confusion[source][predicted] += 1
        total += 1
        if source == predicted:
            correct += 1

    # Per-model accuracy
    model_accuracies = {}
    for fid in facility_ids:
        model_total = sum(confusion[fid].values())
        if model_total > 0:
            model_accuracies[fid] = confusion[fid][fid] / model_total
        else:
            model_accuracies[fid] = 0.0

    overall_accuracy = correct / total if total > 0 else 0.0

    return CrossSiteMetrics(
        model_accuracies=model_accuracies,
        confusion_matrix=confusion,
        overall_accuracy=overall_accuracy,
        inter_facility_lpips=float(np.mean(lpips_inter)) if lpips_inter else 0.0,
        intra_facility_lpips=float(np.mean(lpips_intra)) if lpips_intra else 0.0,
    )

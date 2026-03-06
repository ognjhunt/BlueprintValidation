from __future__ import annotations

import sys
import types


def test_paired_ttest_p_value_equal_vectors_returns_one():
    from blueprint_validation.evaluation.stats_utils import paired_ttest_p_value

    assert paired_ttest_p_value([1.0, 2.0], [1.0, 2.0]) == 1.0


def test_paired_ttest_p_value_constant_offset_returns_zero():
    from blueprint_validation.evaluation.stats_utils import paired_ttest_p_value

    assert paired_ttest_p_value([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == 0.0


def test_paired_ttest_p_value_mismatched_lengths_returns_none():
    from blueprint_validation.evaluation.stats_utils import paired_ttest_p_value

    assert paired_ttest_p_value([1.0, 2.0], [1.0]) is None


def test_paired_ttest_p_value_normalizes_nan(monkeypatch):
    from blueprint_validation.evaluation.stats_utils import paired_ttest_p_value

    scipy = types.ModuleType("scipy")
    scipy.stats = types.SimpleNamespace(ttest_rel=lambda *_a, **_k: (0.0, float("nan")))
    monkeypatch.setitem(sys.modules, "scipy", scipy)

    assert paired_ttest_p_value([1.0, 2.0], [1.5, 2.5]) is None

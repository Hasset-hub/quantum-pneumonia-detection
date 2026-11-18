import numpy as np

from qpd.data import stratified_fraction
from qpd.metrics import binary_metrics


def test_stratified_fraction_keeps_class_ratio_and_no_duplicates():
    labels = np.array([0] * 260 + [1] * 740)
    idx = stratified_fraction(labels, 0.1, seed=0)
    assert len(idx) == len(set(idx)) == 100
    assert abs(labels[idx].mean() - 0.74) < 0.02


def test_tiny_fraction_keeps_both_classes():
    labels = np.array([0] * 30 + [1] * 70)
    assert set(labels[stratified_fraction(labels, 0.001, seed=1)]) == {0, 1}


def test_full_fraction_returns_everything():
    assert len(stratified_fraction(np.zeros(10), 1.0, seed=0)) == 10


def test_binary_metrics_perfect():
    m = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auroc"] == m["accuracy"] == m["sensitivity"] == m["specificity"] == 1.0

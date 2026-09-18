from __future__ import annotations

import numpy as np
import pytest

from floodrisk.model.evaluate import (
    THRESHOLD_GRID,
    EvaluateError,
    best_threshold,
    counts_at,
    dice_at,
    metrics_at,
    per_patch_dice,
)


def single(prob, label, valid=None):
    """Um patch 1 x N, no formato (patches, altura, largura)."""
    probability = np.asarray(prob, dtype="float32").reshape(1, 1, -1)
    reference = np.asarray(label, dtype="float32").reshape(1, 1, -1)
    validity = (
        np.ones_like(reference)
        if valid is None
        else np.asarray(valid, dtype="float32").reshape(1, 1, -1)
    )
    return probability, reference, validity


# --------------------------------------------------------------------------- #
# Contagens
# --------------------------------------------------------------------------- #


def test_a_perfect_prediction_has_no_errors():
    prob, label, valid = single([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0])
    assert counts_at(prob, label, valid, 0.5) == (2.0, 0.0, 0.0, 2.0)


def test_the_counts_add_up_to_the_valid_pixels():
    rng = np.random.default_rng(0)
    prob = rng.random((3, 8, 8)).astype("float32")
    label = (rng.random((3, 8, 8)) > 0.5).astype("float32")
    valid = (rng.random((3, 8, 8)) > 0.2).astype("float32")
    assert sum(counts_at(prob, label, valid, 0.5)) == np.count_nonzero(valid)


def test_invalid_pixels_are_counted_nowhere():
    """Pixel sem dado não pode virar acerto grátis nem erro inventado."""
    prob, label, valid = single([0.9, 0.9], [0, 0], [1, 0])
    assert counts_at(prob, label, valid, 0.5) == (0.0, 1.0, 0.0, 0.0)


def test_the_threshold_is_inclusive_at_its_own_value():
    prob, label, valid = single([0.5], [1])
    assert counts_at(prob, label, valid, 0.5)[0] == 1.0


def test_raising_the_threshold_never_adds_positives():
    rng = np.random.default_rng(1)
    prob = rng.random((2, 6, 6)).astype("float32")
    label = (rng.random((2, 6, 6)) > 0.5).astype("float32")
    valid = np.ones_like(label)
    previous = None
    for threshold in (0.2, 0.4, 0.6, 0.8):
        tp, fp, _, _ = counts_at(prob, label, valid, threshold)
        if previous is not None:
            assert tp + fp <= previous
        previous = tp + fp


# --------------------------------------------------------------------------- #
# Calibração do limiar
# --------------------------------------------------------------------------- #


def test_the_grid_stays_inside_the_useful_range():
    assert min(THRESHOLD_GRID) == pytest.approx(0.05)
    assert max(THRESHOLD_GRID) == pytest.approx(0.95)


def test_the_search_finds_the_separating_threshold():
    """Positivos acima de 0,7 e negativos abaixo de 0,3: qualquer corte no meio
    é perfeito, e o critério de desempate escolhe o mais baixo."""
    prob, label, valid = single([0.8, 0.9, 0.2, 0.1], [1, 1, 0, 0])
    threshold, dice = best_threshold(prob, label, valid)
    assert dice == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.8


def test_ties_favour_recall():
    prob, label, valid = single([0.8, 0.9, 0.2, 0.1], [1, 1, 0, 0])
    threshold, _ = best_threshold(prob, label, valid)
    other = [
        value
        for value in THRESHOLD_GRID
        if dice_at(prob, label, valid, value) == pytest.approx(1.0)
    ]
    assert threshold == min(other)


def test_the_chosen_threshold_is_at_least_as_good_as_any_other():
    rng = np.random.default_rng(2)
    prob = rng.random((4, 10, 10)).astype("float32")
    label = (prob + rng.normal(0, 0.2, prob.shape) > 0.5).astype("float32")
    valid = np.ones_like(label)
    threshold, dice = best_threshold(prob, label, valid)
    assert all(dice_at(prob, label, valid, value) <= dice + 1e-12 for value in THRESHOLD_GRID)
    assert dice_at(prob, label, valid, threshold) == pytest.approx(dice)


def test_an_empty_grid_is_rejected():
    prob, label, valid = single([0.5], [1])
    with pytest.raises(EvaluateError, match="grade"):
        best_threshold(prob, label, valid, grid=[])


# --------------------------------------------------------------------------- #
# Dice por patch
# --------------------------------------------------------------------------- #


def test_one_score_per_patch():
    prob = np.zeros((5, 4, 4), dtype="float32")
    assert per_patch_dice(prob, prob, np.ones_like(prob), 0.5).shape == (5,)


def test_an_empty_patch_predicted_empty_scores_one():
    """Acertar 'não há nada aqui' é acerto — devolver 0 colocaria os acertos
    perfeitos no topo da lista de piores, que é justamente a que vai para a figura."""
    prob = np.zeros((1, 4, 4), dtype="float32")
    label = np.zeros((1, 4, 4), dtype="float32")
    assert per_patch_dice(prob, label, np.ones_like(prob), 0.5)[0] == 1.0


def test_patches_are_scored_independently():
    prob = np.stack([np.ones((2, 2)), np.zeros((2, 2))]).astype("float32")
    label = np.ones((2, 2, 2), dtype="float32")
    scores = per_patch_dice(prob, label, np.ones_like(prob), 0.5)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_the_worst_patches_sort_to_the_front():
    prob = np.stack([np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2))]).astype("float32")
    label = np.ones((3, 2, 2), dtype="float32")
    scores = per_patch_dice(prob, label, np.ones_like(prob), 0.5)
    assert int(np.argsort(scores)[0]) == 1


# --------------------------------------------------------------------------- #
# Painel de métricas
# --------------------------------------------------------------------------- #


def test_the_panel_carries_the_raw_confusion_matrix():
    prob, label, valid = single([0.9, 0.1, 0.9, 0.1], [1, 1, 0, 0])
    metrics = metrics_at(prob, label, valid, 0.5)
    assert (metrics["true_positive"], metrics["false_negative"]) == (1.0, 1.0)
    assert (metrics["false_positive"], metrics["true_negative"]) == (1.0, 1.0)


def test_accuracy_and_specificity_are_reported():
    prob, label, valid = single([0.9, 0.1, 0.9, 0.1], [1, 1, 0, 0])
    metrics = metrics_at(prob, label, valid, 0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)


def test_the_panel_records_the_threshold_it_used():
    prob, label, valid = single([0.9], [1])
    assert metrics_at(prob, label, valid, 0.37)["threshold"] == pytest.approx(0.37)


def test_the_positive_rate_describes_the_reference_not_the_prediction():
    """É a prevalência do conjunto — a informação que torna o Dice comparável."""
    prob, label, valid = single([0.0, 0.0, 0.0, 0.0], [1, 1, 1, 0])
    assert metrics_at(prob, label, valid, 0.5)["positive_rate"] == pytest.approx(0.75)


def test_a_fully_invalid_patch_does_not_divide_by_zero():
    prob, label, valid = single([0.9, 0.9], [1, 1], [0, 0])
    metrics = metrics_at(prob, label, valid, 0.5)
    assert metrics["dice"] == 0.0 and metrics["accuracy"] == 0.0

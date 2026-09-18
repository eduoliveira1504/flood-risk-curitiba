from __future__ import annotations

import csv

import numpy as np
import pytest
import yaml

from floodrisk import artifacts
from floodrisk.config import find_repo_root, load_config
from floodrisk.model.data import (
    REFLECTANCE_SCALE,
    DataError,
    Normalisation,
    PatchDataset,
    compute_normalisation,
    load_manifest,
)
from floodrisk.model.loss import segmentation_metrics


def raw_config():
    with (find_repo_root() / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def repo(tmp_path):
    """Repositório de brinquedo com um manifesto e patches .npz sintéticos."""
    (tmp_path / "configs").mkdir()
    with (tmp_path / "configs" / "default.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config(), handle)
    cfg = load_config(root=tmp_path)
    cfg.ensure_dirs()
    return cfg


def write_patch(config, split, name, image, mask, valid):
    folder = artifacts.patches_dir(config) / split
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{name}.npz"
    np.savez_compressed(destination, image=image, mask=mask, valid=valid)
    return {
        "id": name,
        "split": split,
        "file": str(destination.relative_to(config.root)),
    }


def write_manifest(config, rows):
    manifest = artifacts.patch_manifest(config)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "split", "file"])
        writer.writeheader()
        writer.writerows(rows)


def simple_patch(value, size=8, bands=4, valid_all=True):
    image = np.full((bands, size, size), value, dtype="uint16")
    mask = np.zeros((size, size), dtype="uint8")
    mask[: size // 2] = 1
    valid = np.ones((size, size), dtype="uint8")
    if not valid_all:
        valid[:, : size // 2] = 0
    return image, mask, valid


# --------------------------------------------------------------------------- #
# Manifesto
# --------------------------------------------------------------------------- #


def test_manifest_can_be_filtered_by_split(repo):
    rows = [
        write_patch(repo, "train", "a", *simple_patch(1000)),
        write_patch(repo, "val", "b", *simple_patch(2000)),
    ]
    write_manifest(repo, rows)
    assert len(load_manifest(repo, "train")) == 1
    assert len(load_manifest(repo)) == 2


def test_a_missing_manifest_points_at_the_stage_that_writes_it(repo):
    with pytest.raises(DataError, match="make-dataset"):
        load_manifest(repo)


def test_an_unknown_split_lists_the_ones_that_exist(repo):
    write_manifest(repo, [write_patch(repo, "train", "a", *simple_patch(1000))])
    with pytest.raises(DataError, match="train"):
        load_manifest(repo, "teste")


# --------------------------------------------------------------------------- #
# Normalização
# --------------------------------------------------------------------------- #


def test_statistics_are_in_reflectance_not_in_raw_counts(repo):
    rows = [write_patch(repo, "train", "a", *simple_patch(2000))]
    stats = compute_normalisation(repo, rows)
    assert stats.mean[0] == pytest.approx(2000 / REFLECTANCE_SCALE)


def test_statistics_ignore_invalid_pixels(repo):
    """Metade do patch é nodata com valor zero; contá-la puxaria a média para baixo."""
    image, mask, valid = simple_patch(3000, valid_all=False)
    image[:, :, : image.shape[2] // 2] = 0
    rows = [write_patch(repo, "train", "a", image, mask, valid)]
    stats = compute_normalisation(repo, rows)
    assert stats.mean[0] == pytest.approx(3000 / REFLECTANCE_SCALE)


def test_statistics_pool_every_patch(repo):
    rows = [
        write_patch(repo, "train", "a", *simple_patch(1000)),
        write_patch(repo, "train", "b", *simple_patch(3000)),
    ]
    stats = compute_normalisation(repo, rows)
    assert stats.mean[0] == pytest.approx(2000 / REFLECTANCE_SCALE)


def test_a_constant_band_gets_a_non_negative_deviation(repo):
    stats = compute_normalisation(repo, [write_patch(repo, "train", "a", *simple_patch(1500))])
    assert all(value >= 0 for value in stats.std)


def test_statistics_without_any_valid_pixel_are_rejected(repo):
    image, mask, _ = simple_patch(1000)
    valid = np.zeros(mask.shape, dtype="uint8")
    rows = [write_patch(repo, "train", "a", image, mask, valid)]
    with pytest.raises(DataError, match="nenhum pixel válido"):
        compute_normalisation(repo, rows)


def test_statistics_round_trip_through_json():
    stats = Normalisation(mean=[0.1, 0.2], std=[0.3, 0.4])
    assert Normalisation.from_dict(stats.as_dict()) == stats


# --------------------------------------------------------------------------- #
# Item do dataset
# --------------------------------------------------------------------------- #


def dataset(repo, rows, stats=None, augment=False):
    stats = stats or compute_normalisation(repo, rows)
    return PatchDataset(repo, rows, stats, augment=augment)


def test_item_shapes_match_what_the_network_expects(repo):
    rows = [write_patch(repo, "train", "a", *simple_patch(2000))]
    image, label, valid = dataset(repo, rows).load_arrays(0)
    assert image.shape == (4, 8, 8)
    assert label.shape == (1, 8, 8)
    assert valid.shape == (1, 8, 8)


def test_the_image_is_standardised(repo):
    rows = [
        write_patch(repo, "train", "a", *simple_patch(1000)),
        write_patch(repo, "train", "b", *simple_patch(3000)),
    ]
    data = dataset(repo, rows)
    low = data.load_arrays(0)[0]
    high = data.load_arrays(1)[0]
    # Média 2000, então um patch fica abaixo e o outro acima de zero.
    assert low.mean() < 0 < high.mean()


def test_invalid_pixels_enter_the_network_as_zero(repo):
    image, mask, valid = simple_patch(3000, valid_all=False)
    rows = [write_patch(repo, "train", "a", image, mask, valid)]
    data, _, validity = dataset(repo, rows).load_arrays(0)
    assert np.all(data[:, validity[0] == 0] == 0)


def test_the_label_stays_binary(repo):
    rows = [write_patch(repo, "train", "a", *simple_patch(2000))]
    _, label, _ = dataset(repo, rows).load_arrays(0)
    assert set(np.unique(label)) <= {0.0, 1.0}


def test_an_empty_split_is_rejected(repo):
    with pytest.raises(DataError, match="vazio"):
        PatchDataset(repo, [], Normalisation(mean=[1.0], std=[1.0]))


def test_a_zero_deviation_is_rejected(repo):
    rows = [write_patch(repo, "train", "a", *simple_patch(2000))]
    with pytest.raises(DataError, match="desvio padrão"):
        PatchDataset(repo, rows, Normalisation(mean=[0.0] * 4, std=[0.0] * 4))


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #


def asymmetric_patch(size=8, bands=4):
    """Padrão sem nenhuma simetria, para que qualquer giro seja detectável."""
    base = np.arange(size * size, dtype="uint16").reshape(size, size) * 7 + 500
    image = np.stack([base + 100 * b for b in range(bands)])
    mask = (base % 3 == 0).astype("uint8")
    valid = np.ones((size, size), dtype="uint8")
    return image, mask, valid


def test_augmentation_moves_imagery_and_label_together(repo):
    """Girar a imagem sem girar o rótulo destrói o treino em silêncio."""
    rows = [write_patch(repo, "train", "a", *asymmetric_patch())]
    plain = dataset(repo, rows, augment=False)
    augmented = dataset(repo, rows, stats=compute_normalisation(repo, rows), augment=True)

    reference_image, reference_label, _ = plain.load_arrays(0)
    np.random.seed(0)
    for _ in range(12):
        image, label, _ = augmented.load_arrays(0)
        # Descobre qual das oito simetrias saiu comparando a imagem, e exige que
        # o rótulo tenha sofrido exatamente a mesma.
        for turns in range(4):
            for flip in (False, True):
                candidate = np.rot90(reference_image, turns, axes=(-2, -1))
                expected_label = np.rot90(reference_label, turns, axes=(-2, -1))
                if flip:
                    candidate = candidate[..., ::-1]
                    expected_label = expected_label[..., ::-1]
                if np.allclose(candidate, image):
                    assert np.array_equal(expected_label, label)
                    break
            else:
                continue
            break
        else:
            raise AssertionError("a augmentation não é uma simetria do quadrado")


def test_augmentation_preserves_the_set_of_values(repo):
    """D4 permuta pixels; nenhum valor novo pode aparecer."""
    rows = [write_patch(repo, "train", "a", *asymmetric_patch())]
    plain = dataset(repo, rows, augment=False).load_arrays(0)[0]
    augmented = dataset(
        repo, rows, stats=compute_normalisation(repo, rows), augment=True
    )
    np.random.seed(3)
    expected = np.sort(plain, axis=None)
    for _ in range(8):
        assert np.allclose(np.sort(augmented.load_arrays(0)[0], axis=None), expected)


def test_augmentation_actually_changes_something(repo):
    rows = [write_patch(repo, "train", "a", *asymmetric_patch())]
    augmented = dataset(repo, rows, augment=True)
    np.random.seed(1)
    seen = {augmented.load_arrays(0)[0].tobytes() for _ in range(30)}
    assert len(seen) > 1


def test_without_augmentation_the_item_is_stable(repo):
    rows = [write_patch(repo, "train", "a", *asymmetric_patch())]
    data = dataset(repo, rows, augment=False)
    first = data.load_arrays(0)[0]
    for _ in range(5):
        assert np.array_equal(data.load_arrays(0)[0], first)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #


def test_a_perfect_prediction_scores_one():
    assert segmentation_metrics(100, 0, 0) == {
        "dice": 1.0,
        "iou": 1.0,
        "precision": 1.0,
        "recall": 1.0,
    }


def test_dice_is_never_below_iou():
    for tp, fp, fn in [(10, 5, 5), (1, 9, 9), (50, 1, 20)]:
        metrics = segmentation_metrics(tp, fp, fn)
        assert metrics["dice"] >= metrics["iou"]


def test_predicting_nothing_scores_zero():
    metrics = segmentation_metrics(0, 0, 30)
    assert metrics["dice"] == 0.0 and metrics["recall"] == 0.0


def test_empty_counts_do_not_divide_by_zero():
    assert segmentation_metrics(0, 0, 0) == {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }


def test_dice_matches_the_f1_definition():
    metrics = segmentation_metrics(30, 10, 20)
    harmonic = (
        2
        * metrics["precision"]
        * metrics["recall"]
        / (metrics["precision"] + metrics["recall"])
    )
    assert metrics["dice"] == pytest.approx(harmonic)

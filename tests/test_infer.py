from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from floodrisk.config import Config, ConfigError, find_repo_root, load_config
from floodrisk.geo import patch_windows
from floodrisk.model.infer import WEIGHT_FLOOR, InferError, blend_weights


@pytest.fixture
def config():
    return load_config()


def raw_config():
    with (find_repo_root() / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build(data):
    return Config.from_dict(data, root=find_repo_root())


# --------------------------------------------------------------------------- #
# Pesos de mistura
# --------------------------------------------------------------------------- #


def test_the_window_matches_the_patch():
    assert blend_weights(128).shape == (128, 128)


def test_the_centre_weighs_more_than_the_corner():
    """É o ponto inteiro: o pixel central viu contexto completo, o do canto não."""
    weights = blend_weights(64)
    assert weights[32, 32] > weights[0, 0]


def test_no_weight_is_zero():
    """Pixel na borda do raster pode ser coberto só por esta janela, e só nela."""
    assert blend_weights(32).min() > 0


def test_the_floor_is_respected():
    weights = blend_weights(48)
    assert weights.min() == pytest.approx(WEIGHT_FLOOR * weights.max())


def test_the_window_is_symmetric():
    weights = blend_weights(40)
    assert np.allclose(weights, weights[::-1])
    assert np.allclose(weights, weights[:, ::-1])
    assert np.allclose(weights, weights.T)


def test_a_degenerate_window_is_rejected():
    with pytest.raises(InferError, match="2 pixels"):
        blend_weights(1)


# --------------------------------------------------------------------------- #
# Cobertura: a propriedade que o 'infer' depende para não dividir por zero
# --------------------------------------------------------------------------- #


def reconstruct(width, height, patch, overlap):
    """Simula o acumulador de pesos do estágio, sem tocar em PyTorch."""
    weights = blend_weights(patch)
    total = np.zeros((height, width), dtype="float64")
    for window in patch_windows(width, height, patch, overlap):
        total[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ] += weights
    return total


def test_every_pixel_is_covered_by_at_least_one_window(config):
    total = reconstruct(600, 400, config.raster.patch_size, config.raster.inference_overlap)
    assert total.min() > 0


def test_the_interior_is_covered_several_times(config):
    """Se o miolo fosse coberto uma vez só, a média ponderada não faria nada."""
    patch = config.raster.patch_size
    total = reconstruct(600, 400, patch, config.raster.inference_overlap)
    weights = blend_weights(patch)
    assert total[200, 300] > weights.max()


def test_coverage_holds_on_a_raster_barely_larger_than_the_patch(config):
    patch = config.raster.patch_size
    total = reconstruct(patch + 3, patch + 3, patch, config.raster.inference_overlap)
    assert total.min() > 0


def test_a_constant_prediction_survives_the_blend(config):
    """Predizer 0,7 em toda parte tem de reconstruir 0,7 em toda parte — senão a
    mistura está introduzindo viés espacial em vez de só costurar."""
    patch = config.raster.patch_size
    overlap = config.raster.inference_overlap
    weights = blend_weights(patch)
    width, height = 500, 380

    accumulated = np.zeros((height, width), dtype="float64")
    total = np.zeros((height, width), dtype="float64")
    for window in patch_windows(width, height, patch, overlap):
        rows = slice(window.row_off, window.row_off + window.height)
        cols = slice(window.col_off, window.col_off + window.width)
        accumulated[rows, cols] += 0.7 * weights
        total[rows, cols] += weights

    assert np.allclose(accumulated / total, 0.7)


# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #


def test_the_shipped_overlap_exceeds_the_training_one(config):
    assert config.raster.inference_overlap > config.raster.patch_overlap


def test_an_overlap_as_large_as_the_patch_is_rejected():
    data = copy.deepcopy(raw_config())
    data["raster"]["inference_overlap"] = data["raster"]["patch_size"]
    with pytest.raises(ConfigError, match="inference_overlap"):
        build(data)


def test_an_inference_overlap_below_the_training_one_is_rejected():
    data = copy.deepcopy(raw_config())
    data["raster"]["inference_overlap"] = 0
    with pytest.raises(ConfigError, match="inference_overlap"):
        build(data)

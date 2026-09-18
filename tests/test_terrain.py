from __future__ import annotations

import numpy as np
import pytest

from floodrisk.acquisition.dem import (
    DEMError,
    tile_name,
    tile_url,
    tiles_for_bounds,
    validate_elevation,
)
from floodrisk.config import load_config
from floodrisk.features.terrain import TerrainError, slope_degrees


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def terrain(config):
    return config.terrain


# --------------------------------------------------------------------------- #
# Nomenclatura dos tiles
# --------------------------------------------------------------------------- #


def test_curitiba_falls_in_s26_w050():
    assert tile_name(-25.4284, -49.2733) == "S26_00_W050_00"


def test_south_and_west_use_the_floor_not_truncation():
    """Truncar daria S25/W049 — tile vizinho, e o erro passaria despercebido."""
    assert tile_name(-25.43, -49.27) == "S26_00_W050_00"
    assert tile_name(-0.5, -0.5) == "S01_00_W001_00"


def test_northern_and_eastern_hemispheres():
    assert tile_name(0.0, 0.0) == "N00_00_E000_00"
    assert tile_name(47.3, 8.5) == "N47_00_E008_00"


def test_exact_corner_belongs_to_its_own_tile():
    assert tile_name(-26.0, -50.0) == "S26_00_W050_00"


def test_longitude_uses_three_digits_and_latitude_two():
    assert tile_name(-9.0, -9.0) == "S09_00_W009_00"
    assert tile_name(-9.0, -120.0) == "S09_00_W120_00"


def test_curitiba_bbox_needs_a_single_tile(config):
    assert tiles_for_bounds(config.aoi.bbox) == ["S26_00_W050_00"]


def test_envelope_crossing_a_degree_needs_two_tiles():
    assert len(tiles_for_bounds((-49.5, -25.5, -48.5, -25.2))) == 2


def test_degenerate_envelope_is_rejected():
    with pytest.raises(DEMError, match="degenerado"):
        tiles_for_bounds((0, 0, 0, 0))


def test_url_has_no_leftover_placeholder(terrain):
    url = tile_url(terrain, "S26_00_W050_00")
    assert url.endswith("Copernicus_DSM_COG_10_S26_00_W050_00_DEM.tif")
    assert "{" not in url


# --------------------------------------------------------------------------- #
# Validação de altitude
# --------------------------------------------------------------------------- #


def test_curitiba_altitudes_pass(terrain):
    validate_elevation(np.array([880.0, 930.0, 1010.0], dtype="float32"), terrain)


def test_sea_level_is_rejected(terrain):
    """Altitude de litoral significa que veio o tile errado."""
    with pytest.raises(DEMError, match="fora da faixa"):
        validate_elevation(np.array([0.0, 5.0, 12.0], dtype="float32"), terrain)


def test_himalayan_altitudes_are_rejected(terrain):
    with pytest.raises(DEMError, match="fora da faixa"):
        validate_elevation(np.array([900.0, 8000.0], dtype="float32"), terrain)


def test_all_nan_is_rejected(terrain):
    with pytest.raises(DEMError, match="nenhum pixel válido"):
        validate_elevation(np.full(9, np.nan, dtype="float32"), terrain)


def test_nodata_sentinel_is_ignored(terrain):
    from floodrisk.acquisition.dem import NODATA

    validate_elevation(np.array([NODATA, 900.0, 950.0], dtype="float32"), terrain)


def test_inverted_range_is_rejected_by_the_config():
    import copy

    import yaml

    from floodrisk.config import Config, ConfigError, find_repo_root

    with (find_repo_root() / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    data = copy.deepcopy(raw)
    data["terrain"]["expected_elevation_range_m"] = [1200.0, 700.0]
    with pytest.raises(ConfigError, match="expected_elevation_range_m"):
        Config.from_dict(data, root=find_repo_root())


# --------------------------------------------------------------------------- #
# Declividade (Horn)
# --------------------------------------------------------------------------- #


def test_flat_terrain_has_zero_slope():
    assert slope_degrees(np.full((5, 5), 900.0), 10.0, 10.0) == pytest.approx(0.0)


def test_forty_five_degree_ramp():
    """Subida de 10 m a cada 10 m de distância é exatamente 45 graus."""
    ramp = np.tile(np.arange(5, dtype="float64") * 10.0, (5, 1))
    slope = slope_degrees(ramp, 10.0, 10.0)
    # O interior é exato; a borda replicada achata a derivada de propósito.
    assert slope[2, 2] == pytest.approx(45.0, abs=1e-4)


def test_slope_is_independent_of_ramp_direction():
    """A declividade usa a magnitude do gradiente — subir ou descer dá o mesmo."""
    up = np.tile(np.arange(6, dtype="float64") * 5.0, (6, 1))
    down = up[:, ::-1].copy()
    assert slope_degrees(up, 10.0, 10.0)[3, 3] == pytest.approx(
        slope_degrees(down, 10.0, 10.0)[3, 3]
    )


def test_horizontal_and_vertical_ramps_match():
    ramp = np.tile(np.arange(6, dtype="float64") * 5.0, (6, 1))
    assert slope_degrees(ramp, 10.0, 10.0)[3, 3] == pytest.approx(
        slope_degrees(ramp.T, 10.0, 10.0)[3, 3]
    )


def test_larger_cellsize_gives_gentler_slope():
    """Mesmo desnível espalhado por célula maior é rampa mais suave."""
    ramp = np.tile(np.arange(6, dtype="float64") * 10.0, (6, 1))
    fine = slope_degrees(ramp, 10.0, 10.0)[3, 3]
    coarse = slope_degrees(ramp, 30.0, 30.0)[3, 3]
    assert coarse < fine


def test_output_keeps_the_input_shape():
    assert slope_degrees(np.random.default_rng(0).normal(900, 5, (17, 23)), 10.0, 10.0).shape == (
        17,
        23,
    )


def test_slope_is_never_negative():
    noisy = np.random.default_rng(1).normal(900, 20, (30, 30))
    assert (slope_degrees(noisy, 10.0, 10.0) >= 0).all()


def test_slope_never_reaches_ninety_degrees():
    """arctan é assintótico: nem um paredão vertical chega a 90."""
    cliff = np.zeros((5, 5))
    cliff[:, 3:] = 10_000.0
    assert slope_degrees(cliff, 10.0, 10.0).max() < 90.0


def test_non_positive_cellsize_is_rejected():
    with pytest.raises(TerrainError, match="célula"):
        slope_degrees(np.zeros((4, 4)), 0.0, 10.0)


def test_one_dimensional_input_is_rejected():
    with pytest.raises(TerrainError, match="2D"):
        slope_degrees(np.zeros(9), 10.0, 10.0)


def test_array_smaller_than_the_window_is_rejected():
    with pytest.raises(TerrainError, match="3x3"):
        slope_degrees(np.zeros((2, 2)), 10.0, 10.0)


def test_horn_weights_orthogonal_neighbours_double():
    """Sinal de Horn: um pico isolado tem declividade menor que um degrau em linha.

    É essa ponderação que amortece ruído pontual do DEM sem borrar quebra real.
    """
    spike = np.zeros((5, 5))
    spike[2, 3] = 30.0

    step = np.zeros((5, 5))
    step[:, 3] = 30.0

    assert slope_degrees(spike, 10.0, 10.0)[2, 2] < slope_degrees(step, 10.0, 10.0)[2, 2]

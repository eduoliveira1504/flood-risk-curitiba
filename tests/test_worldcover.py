from __future__ import annotations

import numpy as np
import pytest

from floodrisk.acquisition.worldcover import (
    WORLDCOVER_CLASSES,
    WorldCoverError,
    tile_name,
    tile_url,
    tiles_for_bounds,
    validate_classes,
)
from floodrisk.config import load_config


@pytest.fixture
def config():
    return load_config()


# --------------------------------------------------------------------------- #
# Nomenclatura dos tiles — onde mora o bug clássico
# --------------------------------------------------------------------------- #


def test_curitiba_falls_in_s27w051():
    """Curitiba está em -25.43, -49.27. O tile é nomeado pelo canto SUDOESTE."""
    assert tile_name(-25.4284, -49.2733) == "S27W051"


def test_south_and_west_round_down_not_toward_zero():
    """Arredondar para zero é o erro clássico e só aparece em S/W — ou seja, aqui.

    -25.43 / 3 = -8.48. Truncando dá -8 (tile S24, ERRADO); o piso dá -9 (S27).
    """
    assert tile_name(-25.43, -49.27) == "S27W051"
    assert tile_name(-1.0, -1.0) == "S03W003"
    assert tile_name(-0.1, -0.1) == "S03W003"


def test_northern_and_eastern_hemispheres():
    assert tile_name(0.0, 0.0) == "N00E000"
    assert tile_name(2.9, 2.9) == "N00E000"
    assert tile_name(3.0, 3.0) == "N03E003"
    assert tile_name(51.5, -0.12) == "N51W003"


def test_exact_tile_corner_belongs_to_its_own_tile():
    assert tile_name(-27.0, -51.0) == "S27W051"
    assert tile_name(-24.0, -48.0) == "S24W048"


def test_longitude_is_zero_padded_to_three_digits():
    assert tile_name(-25.0, -9.0) == "S27W009"
    assert tile_name(-25.0, -120.0) == "S27W120"


def test_tile_name_rejects_bad_size():
    with pytest.raises(WorldCoverError, match="tile_size_deg"):
        tile_name(0, 0, 0)


# --------------------------------------------------------------------------- #
# Cobertura do envelope
# --------------------------------------------------------------------------- #


def test_curitiba_bbox_needs_a_single_tile(config):
    tiles = tiles_for_bounds(config.aoi.bbox, config.ground_truth.worldcover.tile_size_deg)
    assert tiles == ["S27W051"]


def test_envelope_spanning_two_tiles_returns_both():
    # Cruza a longitude -48, que é fronteira de tile.
    tiles = tiles_for_bounds((-49.0, -25.0, -47.0, -24.5))
    assert set(tiles) == {"S27W051", "S27W048"}


def test_envelope_spanning_four_tiles():
    tiles = tiles_for_bounds((-49.0, -25.0, -47.0, -23.0))
    assert len(tiles) == 4


def test_no_duplicate_tiles():
    tiles = tiles_for_bounds((-50.9, -26.9, -48.1, -24.1))
    assert len(tiles) == len(set(tiles))


def test_degenerate_envelope_is_rejected():
    with pytest.raises(WorldCoverError, match="degenerado"):
        tiles_for_bounds((0, 0, 0, 0))


# --------------------------------------------------------------------------- #
# URL
# --------------------------------------------------------------------------- #


def test_url_is_built_from_the_template(config):
    url = tile_url(config.ground_truth.worldcover, "S27W051")
    assert url.endswith("ESA_WorldCover_10m_2021_v200_S27W051_Map.tif")
    assert url.startswith("https://")
    assert "{" not in url, "sobrou marcador não substituído"


# --------------------------------------------------------------------------- #
# Validação do conteúdo
# --------------------------------------------------------------------------- #


def test_legend_codes_pass():
    validate_classes(np.array(list(WORLDCOVER_CLASSES), dtype="uint8"))


def test_code_outside_the_legend_is_rejected():
    """Se o arquivo baixado não for WorldCover, os códigos denunciam."""
    with pytest.raises(WorldCoverError, match="fora da legenda"):
        validate_classes(np.array([10, 50, 137], dtype="uint8"))


def test_empty_crop_is_rejected():
    with pytest.raises(WorldCoverError, match="nenhuma classe"):
        validate_classes(np.zeros((10, 10), dtype="uint8"))


def test_builtup_class_is_in_the_legend(config):
    assert config.ground_truth.worldcover.builtup_class in WORLDCOVER_CLASSES


def test_url_template_must_carry_every_placeholder(config):
    import copy

    import yaml

    from floodrisk.config import Config, ConfigError, find_repo_root

    with (find_repo_root() / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    data = copy.deepcopy(raw)
    data["ground_truth"]["worldcover"]["url_template"] = "https://x/{year}.tif"
    with pytest.raises(ConfigError, match="url_template"):
        Config.from_dict(data, root=find_repo_root())

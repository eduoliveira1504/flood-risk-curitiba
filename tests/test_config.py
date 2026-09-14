from __future__ import annotations

import copy

import pytest
import yaml

from floodrisk.config import Config, ConfigError, find_repo_root, load_config


@pytest.fixture(scope="module")
def raw() -> dict:
    root = find_repo_root()
    with (root / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build(raw: dict) -> Config:
    return Config.from_dict(raw, root=find_repo_root())


def test_default_config_loads():
    config = load_config()
    assert config.project.crs_metric == "EPSG:31982"
    assert config.raster.stride == config.raster.patch_size - config.raster.patch_overlap


def test_paths_resolve_under_repo_root():
    config = load_config()
    for path in config.paths.all().values():
        assert path.is_absolute()
        assert config.root in path.parents or path == config.root


def test_unknown_key_is_rejected(raw):
    data = copy.deepcopy(raw)
    data["project"]["nao_existe"] = 1
    with pytest.raises(ConfigError, match="desconhecida"):
        build(data)


def test_missing_section_is_rejected(raw):
    data = copy.deepcopy(raw)
    del data["terrain"]
    with pytest.raises(ConfigError, match="ausente"):
        build(data)


def test_inverted_bbox_is_rejected(raw):
    data = copy.deepcopy(raw)
    data["aoi"]["bbox"] = [-49.18, -25.34, -49.39, -25.65]
    with pytest.raises(ConfigError, match="bbox"):
        build(data)


def test_split_fractions_must_sum_to_one(raw):
    data = copy.deepcopy(raw)
    data["split"]["fractions"] = {"train": 0.7, "val": 0.2, "test": 0.2}
    with pytest.raises(ConfigError, match="somar"):
        build(data)


def test_non_spatial_split_is_rejected(raw):
    data = copy.deepcopy(raw)
    data["split"]["strategy"] = "random"
    with pytest.raises(ConfigError, match="spatial_block"):
        build(data)


def test_overlap_must_be_smaller_than_patch(raw):
    data = copy.deepcopy(raw)
    data["raster"]["patch_overlap"] = 256
    with pytest.raises(ConfigError, match="patch_overlap"):
        build(data)


def test_commercial_sentinel_hub_host_is_rejected(raw):
    """Trocar o CDSE pelo Sentinel Hub comercial é erro silencioso até virar 401."""
    data = copy.deepcopy(raw)
    data["sentinel"]["base_url"] = "https://services.apps.sentinel-hub.com"
    with pytest.raises(ConfigError, match="comercial"):
        build(data)


def test_sirgas_request_crs_is_rejected(raw):
    """EPSG:31982 é o CRS certo para o Brasil e o Process API não aceita."""
    data = copy.deepcopy(raw)
    data["sentinel"]["request_crs"] = "EPSG:31982"
    with pytest.raises(ConfigError, match="Process API"):
        build(data)


def test_utm_zone_boundaries_are_accepted(raw):
    for code in ("EPSG:32601", "EPSG:32660", "EPSG:32701", "EPSG:32760", "EPSG:4326"):
        data = copy.deepcopy(raw)
        data["sentinel"]["request_crs"] = code
        assert build(data).sentinel.request_crs == code


def test_nonexistent_utm_zones_are_rejected(raw):
    for code in ("EPSG:32600", "EPSG:32661", "EPSG:32799"):
        data = copy.deepcopy(raw)
        data["sentinel"]["request_crs"] = code
        with pytest.raises(ConfigError, match="Process API"):
            build(data)


def test_sentinel_dates_must_be_ordered(raw):
    data = copy.deepcopy(raw)
    data["sentinel"]["date_start"] = "2025-01-01"
    data["sentinel"]["date_end"] = "2024-01-01"
    with pytest.raises(ConfigError, match="anterior"):
        build(data)


def test_cloud_cover_out_of_range_is_rejected(raw):
    data = copy.deepcopy(raw)
    data["sentinel"]["max_cloud_cover"] = 150
    with pytest.raises(ConfigError, match="max_cloud_cover"):
        build(data)


def test_empty_attribution_is_rejected(raw):
    """Open-Meteo é CC-BY: publicar sem crédito é violação de licença."""
    data = copy.deepcopy(raw)
    data["forecast"]["attribution"] = "   "
    with pytest.raises(ConfigError, match="attribution"):
        build(data)


def test_scenario_tiers_must_start_at_zero(raw):
    data = copy.deepcopy(raw)
    data["risk_scenarios"]["tiers"][0]["min_mm"] = 5.0
    with pytest.raises(ConfigError, match="0 mm"):
        build(data)


def test_scenario_tiers_must_be_increasing(raw):
    data = copy.deepcopy(raw)
    data["risk_scenarios"]["tiers"][2]["min_mm"] = 10.0
    with pytest.raises(ConfigError, match="crescente"):
        build(data)


def test_scenario_justification_still_pending(raw):
    """Trocar para false só quando os cortes tiverem base em Lohmann & Santos (2021)."""
    config = build(copy.deepcopy(raw))
    assert config.risk_scenarios.justification_pending is True


def test_occurrences_target_still_undefined(raw):
    """Guarda-chuva: quando a fonte do alvo for definida, este teste deve ser trocado."""
    config = build(copy.deepcopy(raw))
    assert config.occurrences.is_defined is False

from __future__ import annotations

import json
import re

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodrisk.acquisition.sentinel import (
    RESPONSE_BANDS,
    SentinelError,
    build_evalscript,
    credentials_from_env,
    estimate_processing_units,
    merge_tiles,
    request_tiles,
)
from floodrisk.config import load_config


@pytest.fixture
def config():
    return load_config()


# --------------------------------------------------------------------------- #
# Evalscript
# --------------------------------------------------------------------------- #


def test_evalscript_declares_every_requested_band(config):
    script = build_evalscript(config.sentinel.bands, config.sentinel.invalid_scl_classes)
    for band in config.sentinel.bands:
        assert f"acc_{band}" in script
        assert f"sample.{band}" in script


def test_evalscript_output_band_count_matches_config(config):
    script = build_evalscript(config.sentinel.bands, config.sentinel.invalid_scl_classes)
    match = re.search(rf'id: "{RESPONSE_BANDS}", bands: (\d+)', script)
    assert match is not None
    assert int(match.group(1)) == len(config.sentinel.bands)


def test_evalscript_is_multitemporal():
    """Sem mosaicking ORBIT o evalscript recebe uma cena só e não há mediana."""
    script = build_evalscript(["B02"], [])
    assert 'mosaicking: "ORBIT"' in script
    assert "//VERSION=3" in script


def test_evalscript_always_requests_scl_and_datamask():
    script = build_evalscript(["B04", "B08"], [8, 9])
    inputs = json.loads(re.search(r"bands: (\[[^\]]*\])", script).group(1))
    assert inputs == ["B04", "B08", "SCL", "dataMask"]


def test_evalscript_embeds_the_cloud_mask_classes():
    script = build_evalscript(["B02"], [0, 1, 3, 8, 9, 10, 11])
    assert "var INVALID_SCL = [0, 1, 3, 8, 9, 10, 11];" in script


def test_evalscript_tracks_observation_count_for_the_mask():
    """Pixel sem nenhuma cena limpa precisa sair marcado, não como zero válido."""
    script = build_evalscript(["B02", "B03"], [8])
    assert "var observations = acc_B02.length;" in script
    assert "observations > 0 ? 1 : 0" in script


def test_evalscript_adapts_to_a_single_band():
    script = build_evalscript(["B08"], [])
    assert re.search(rf'id: "{RESPONSE_BANDS}", bands: 1', script)
    assert "median(acc_B08)" in script


def test_evalscript_rejects_empty_band_list():
    with pytest.raises(SentinelError, match="nenhuma banda"):
        build_evalscript([], [])


# --------------------------------------------------------------------------- #
# Tiles
# --------------------------------------------------------------------------- #


def test_tiles_cover_the_whole_envelope_without_gaps():
    bounds = (0.0, 0.0, 50_000.0, 30_000.0)
    tiles = request_tiles(bounds, resolution_m=10.0, tile_px=2048)

    assert tiles
    covered = sum((t.bounds[2] - t.bounds[0]) * (t.bounds[3] - t.bounds[1]) for t in tiles)
    expected = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    assert covered == pytest.approx(expected)

    assert min(t.bounds[0] for t in tiles) == pytest.approx(bounds[0])
    assert max(t.bounds[2] for t in tiles) == pytest.approx(bounds[2])
    assert max(t.bounds[3] for t in tiles) == pytest.approx(bounds[3])


def test_tiles_never_exceed_the_pixel_limit():
    tiles = request_tiles((0.0, 0.0, 50_000.0, 30_000.0), 10.0, 2048)
    assert all(t.width_px <= 2048 and t.height_px <= 2048 for t in tiles)


def test_tiles_do_not_overlap():
    tiles = request_tiles((0.0, 0.0, 40_960.0, 40_960.0), 10.0, 2048)
    for i, a in enumerate(tiles):
        for b in tiles[i + 1 :]:
            x_overlap = min(a.bounds[2], b.bounds[2]) - max(a.bounds[0], b.bounds[0])
            y_overlap = min(a.bounds[3], b.bounds[3]) - max(a.bounds[1], b.bounds[1])
            assert x_overlap <= 0 or y_overlap <= 0


def test_tile_count_is_exact_when_the_envelope_divides_evenly():
    # 40.960 m / (2048 px × 10 m) = 2 tiles por lado
    tiles = request_tiles((0.0, 0.0, 40_960.0, 40_960.0), 10.0, 2048)
    assert len(tiles) == 4
    assert all(t.width_px == 2048 and t.height_px == 2048 for t in tiles)


def test_indices_are_unique_and_sequential():
    tiles = request_tiles((0.0, 0.0, 50_000.0, 30_000.0), 10.0, 1024)
    assert [t.index for t in tiles] == list(range(len(tiles)))
    assert len({t.name for t in tiles}) == len(tiles)


def test_curitiba_aoi_fits_in_a_handful_of_tiles(config):
    from floodrisk.geo import aoi_geometry

    aoi = aoi_geometry(config, metric=True)
    tiles = request_tiles(
        aoi.bounds, config.raster.resolution_m, config.sentinel.request_tile_px
    )
    # Sanidade: se isso explodir para centenas de tiles, algo está errado na AOI.
    assert 1 <= len(tiles) <= 12


def test_tiles_reject_bad_arguments():
    with pytest.raises(SentinelError, match="tile_px"):
        request_tiles((0, 0, 100, 100), 10.0, 0)
    with pytest.raises(SentinelError, match="resolution_m"):
        request_tiles((0, 0, 100, 100), 0.0, 2048)


# --------------------------------------------------------------------------- #
# Custo em processing units
# --------------------------------------------------------------------------- #


def test_base_request_costs_one_pu():
    """512×512 px, 3 bandas de entrada, 1 amostra = 1 PU, por definição do CDSE."""
    # 1 banda pedida + SCL + dataMask = 3 bandas de entrada.
    assert estimate_processing_units(512 * 512, n_bands=1, n_samples_per_pixel=1) == (
        pytest.approx(1.0)
    )


def test_cost_is_linear_in_the_number_of_scenes():
    """O fator que domina: janela temporal maior custa proporcionalmente mais."""
    one = estimate_processing_units(512 * 512, 1, 1)
    fifty = estimate_processing_units(512 * 512, 1, 50)
    assert fifty == pytest.approx(one * 50)


def test_cost_is_linear_in_area():
    small = estimate_processing_units(512 * 512, 4, 10)
    big = estimate_processing_units(512 * 512 * 4, 4, 10)
    assert big == pytest.approx(small * 4)


def test_cost_counts_scl_and_datamask_as_input_bands():
    """Erra para cima de propósito — melhor gastar menos que o previsto."""
    assert estimate_processing_units(512 * 512, n_bands=4, n_samples_per_pixel=1) == (
        pytest.approx(2.0)  # (4 + 2) / 3
    )


def test_tiny_request_has_a_floor():
    assert estimate_processing_units(10, 1, 1) == pytest.approx(0.01)


def test_zero_samples_still_costs_one_sample():
    assert estimate_processing_units(512 * 512, 1, 0) == pytest.approx(1.0)


def test_curitiba_one_year_stays_under_the_configured_budget(config):
    """Guarda-vida: a janela padrão do YAML não pode estourar a cota sozinha."""
    from floodrisk.geo import aoi_geometry

    aoi = aoi_geometry(config, metric=True)
    tiles = request_tiles(
        aoi.bounds, config.raster.resolution_m, config.sentinel.request_tile_px
    )
    total_px = sum(t.pixels for t in tiles)
    # ~24 datas com nuvem < 20% em um ano sobre Curitiba.
    estimated = estimate_processing_units(total_px, len(config.sentinel.bands), 24)
    assert estimated <= config.sentinel.max_processing_units
    # E a cota mensal inteira precisa comportar mais de uma execução.
    assert estimated < 10_000 / 2


def test_cost_rejects_negative_input():
    with pytest.raises(SentinelError, match="negativos"):
        estimate_processing_units(-1, 4, 10)


# --------------------------------------------------------------------------- #
# Credenciais
# --------------------------------------------------------------------------- #


def test_credentials_are_read_and_stripped():
    client_id, secret = credentials_from_env(
        {"CDSE_CLIENT_ID": "  abc  ", "CDSE_CLIENT_SECRET": "def\n"}
    )
    assert (client_id, secret) == ("abc", "def")


def test_missing_credentials_name_both_variables_and_the_fix():
    with pytest.raises(SentinelError) as excinfo:
        credentials_from_env({})
    message = str(excinfo.value)
    assert "CDSE_CLIENT_ID" in message
    assert "CDSE_CLIENT_SECRET" in message
    assert ".env" in message


def test_whitespace_only_credential_counts_as_missing():
    """Cola do dashboard com espaço sobrando dá 401 genérico — barrar antes."""
    with pytest.raises(SentinelError, match="CDSE_CLIENT_SECRET"):
        credentials_from_env({"CDSE_CLIENT_ID": "abc", "CDSE_CLIENT_SECRET": "   "})


# --------------------------------------------------------------------------- #
# Costura
# --------------------------------------------------------------------------- #


# Canto sudoeste real de Curitiba em UTM 22S, para que a reprojeção entre
# SIRGAS 2000 e WGS84 seja exercitada com coordenadas plausíveis.
CWB_WEST = 661_000.0
CWB_SOUTH = 7_161_160.0


def write_tile(path, bounds, value, count, crs, resolution=10.0):
    west, south, east, north = bounds
    width = round((east - west) / resolution)
    height = round((north - south) / resolution)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="uint16",
        crs=crs,
        transform=from_origin(west, north, resolution, resolution),
        nodata=0,
    ) as dst:
        for band in range(1, count + 1):
            dst.write(np.full((height, width), value, dtype="uint16"), band)
    return path


def test_merge_stitches_tiles_and_reprojects_to_the_analysis_crs(tmp_path, config):
    raw = tmp_path / "sentinel2"
    bands = len(config.sentinel.bands)
    request_crs = config.sentinel.request_crs
    left = (CWB_WEST, CWB_SOUTH, CWB_WEST + 2000, CWB_SOUTH + 2000)
    right = (CWB_WEST + 2000, CWB_SOUTH, CWB_WEST + 4000, CWB_SOUTH + 2000)
    write_tile(raw / "tile_000" / f"{RESPONSE_BANDS}.tif", left, 100, bands, request_crs)
    write_tile(raw / "tile_001" / f"{RESPONSE_BANDS}.tif", right, 200, bands, request_crs)

    aoi = (CWB_WEST, CWB_SOUTH, CWB_WEST + 4000, CWB_SOUTH + 2000)
    out = merge_tiles(raw, tmp_path / "s2_median.tif", aoi, config)

    with rasterio.open(out) as src:
        assert src.count == bands
        assert src.crs.to_string() == config.project.crs_metric
        # O deslocamento de datum move os cantos por decímetros e o alinhamento à
        # grade arredonda para fora, então a saída pode ganhar até um pixel por
        # lado. Isso é desejado: garante cobrir a AOI inteira, nunca menos.
        assert 400 <= src.width <= 401
        assert 200 <= src.height <= 201
        data = src.read(1)
        # Os dois valores sintéticos precisam sobreviver à reprojeção: vizinho
        # mais próximo não pode inventar valor intermediário.
        assert set(np.unique(data)) <= {0, 100, 200}
        assert (data == 100).any() and (data == 200).any()


def test_merge_preserves_values_exactly_no_interpolation(tmp_path, config):
    """Reamostragem por vizinho: nenhum valor novo pode aparecer."""
    raw = tmp_path / "sentinel2"
    bounds = (CWB_WEST, CWB_SOUTH, CWB_WEST + 1000, CWB_SOUTH + 1000)
    write_tile(
        raw / "tile_000" / f"{RESPONSE_BANDS}.tif",
        bounds,
        1234,
        len(config.sentinel.bands),
        config.sentinel.request_crs,
    )
    out = merge_tiles(raw, tmp_path / "m.tif", bounds, config)

    with rasterio.open(out) as src:
        assert set(np.unique(src.read(1))) <= {0, 1234}


def test_merge_records_provenance_in_the_tags(tmp_path, config):
    raw = tmp_path / "sentinel2"
    bounds = (CWB_WEST, CWB_SOUTH, CWB_WEST + 1000, CWB_SOUTH + 1000)
    write_tile(
        raw / "tile_000" / f"{RESPONSE_BANDS}.tif",
        bounds,
        50,
        len(config.sentinel.bands),
        config.sentinel.request_crs,
    )
    out = merge_tiles(raw, tmp_path / "m.tif", bounds, config)

    with rasterio.open(out) as src:
        tags = src.tags()
        assert tags["composite"] == "median"
        assert tags["reflectance_scale"] == "10000"
        assert tags["date_start"] == config.sentinel.date_start
        assert "Sentinel-2" in tags["source"]
        # A proveniência precisa registrar os dois CRS: em qual foi pedido e em
        # qual está o arquivo. Sem isso ninguém reconstrói o caminho depois.
        assert tags["request_crs"] == config.sentinel.request_crs
        assert tags["analysis_crs"] == config.project.crs_metric
        assert list(src.descriptions) == list(config.sentinel.bands)


def test_write_geotiff_georeferences_a_multiband_array(tmp_path, config):
    """tifffile devolve (altura, largura, banda); rasterio espera o inverso."""
    from floodrisk.acquisition.sentinel import write_geotiff

    array = np.zeros((100, 200, 4), dtype="uint16")
    array[:, :, 0] = 7
    bounds = (CWB_WEST, CWB_SOUTH, CWB_WEST + 2000, CWB_SOUTH + 1000)

    out = write_geotiff(array, bounds, "EPSG:32722", 10.0, tmp_path / "bands.tif")

    with rasterio.open(out) as src:
        assert src.count == 4
        assert (src.width, src.height) == (200, 100)
        assert src.crs.to_string() == "EPSG:32722"
        assert src.bounds.left == pytest.approx(CWB_WEST)
        assert src.bounds.top == pytest.approx(CWB_SOUTH + 1000)
        assert (src.read(1) == 7).all()


def test_write_geotiff_handles_a_single_band_mask(tmp_path, config):
    from floodrisk.acquisition.sentinel import write_geotiff

    array = np.ones((50, 60), dtype="uint8")
    bounds = (CWB_WEST, CWB_SOUTH, CWB_WEST + 600, CWB_SOUTH + 500)

    out = write_geotiff(array, bounds, "EPSG:32722", 10.0, tmp_path / "mask.tif")

    with rasterio.open(out) as src:
        assert src.count == 1
        assert (src.width, src.height) == (60, 50)


def test_write_geotiff_rejects_unexpected_dimensions(tmp_path):
    from floodrisk.acquisition.sentinel import write_geotiff

    with pytest.raises(SentinelError, match="dimensões"):
        write_geotiff(
            np.zeros((2, 2, 2, 2), dtype="uint16"),
            (0, 0, 20, 20),
            "EPSG:32722",
            10.0,
            tmp_path / "x.tif",
        )


def test_merge_fails_loudly_when_nothing_was_downloaded(tmp_path, config):
    empty = tmp_path / "sentinel2"
    empty.mkdir()
    with pytest.raises(SentinelError, match="nenhum"):
        merge_tiles(empty, tmp_path / "m.tif", (0, 0, 1000, 1000), config)

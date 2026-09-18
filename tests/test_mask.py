from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from floodrisk.config import load_config
from floodrisk.features.mask import (
    IMPERVIOUS,
    PERVIOUS,
    MaskError,
    build,
    combine,
    summarise,
)

BUILTUP = 50
TREES = 10


@pytest.fixture
def config():
    return load_config()


# --------------------------------------------------------------------------- #
# Regra de decisão
# --------------------------------------------------------------------------- #


def test_worldcover_builtup_alone_marks_impervious():
    worldcover = np.array([[BUILTUP, TREES]], dtype="uint8")
    roads = np.zeros((1, 2), dtype="uint8")
    assert combine(worldcover, BUILTUP, roads).tolist() == [[IMPERVIOUS, PERVIOUS]]


def test_road_alone_marks_impervious():
    """O caso que justifica o OSM: rua que o WorldCover classificou como árvore."""
    worldcover = np.array([[TREES, TREES]], dtype="uint8")
    roads = np.array([[1, 0]], dtype="uint8")
    assert combine(worldcover, BUILTUP, roads).tolist() == [[IMPERVIOUS, PERVIOUS]]


def test_union_not_intersection():
    worldcover = np.array([[BUILTUP, TREES, TREES]], dtype="uint8")
    roads = np.array([[0, 1, 0]], dtype="uint8")
    assert combine(worldcover, BUILTUP, roads).tolist() == [
        [IMPERVIOUS, IMPERVIOUS, PERVIOUS]
    ]


def test_other_worldcover_classes_are_pervious():
    worldcover = np.array([[10, 20, 30, 40, 60, 80, 90]], dtype="uint8")
    roads = np.zeros((1, 7), dtype="uint8")
    assert combine(worldcover, BUILTUP, roads).sum() == 0


def test_result_is_strictly_binary_uint8():
    worldcover = np.array([[BUILTUP, TREES]], dtype="uint8")
    roads = np.array([[1, 1]], dtype="uint8")
    out = combine(worldcover, BUILTUP, roads)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_shape_mismatch_is_rejected():
    with pytest.raises(MaskError, match="incompatíveis"):
        combine(np.zeros((2, 2), "uint8"), BUILTUP, np.zeros((3, 3), "uint8"))


# --------------------------------------------------------------------------- #
# Decomposição por origem
# --------------------------------------------------------------------------- #


def test_summary_splits_contributions_by_source():
    # 4 pixels: só WorldCover, só via, ambos, nenhum.
    worldcover = np.array([[BUILTUP, TREES, BUILTUP, TREES]], dtype="uint8")
    roads = np.array([[0, 1, 1, 0]], dtype="uint8")
    mask = combine(worldcover, BUILTUP, roads)

    stats = summarise(mask, worldcover, BUILTUP, roads)
    assert stats["impervious_pct"] == pytest.approx(75.0)
    assert stats["worldcover_only_pct"] == pytest.approx(25.0)
    assert stats["osm_only_pct"] == pytest.approx(25.0)
    assert stats["both_pct"] == pytest.approx(25.0)


def test_summary_parts_add_up_to_the_total():
    rng = np.random.default_rng(0)
    worldcover = rng.choice([TREES, BUILTUP], size=(40, 40)).astype("uint8")
    roads = rng.choice([0, 1], size=(40, 40)).astype("uint8")
    mask = combine(worldcover, BUILTUP, roads)
    stats = summarise(mask, worldcover, BUILTUP, roads)

    parts = stats["worldcover_only_pct"] + stats["osm_only_pct"] + stats["both_pct"]
    assert parts == pytest.approx(stats["impervious_pct"])


# --------------------------------------------------------------------------- #
# Estágio completo
# --------------------------------------------------------------------------- #


def write_worldcover(path, array, transform, crs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1], count=1,
        dtype="uint8", crs=crs, transform=transform, nodata=0,
    ) as dst:
        dst.write(array, 1)


def prepare(tmp_path, config, worldcover_array, lines, buffer_m=5.0):
    """Monta os dois insumos em disco e aponta a config para eles."""
    from floodrisk import artifacts

    resolution = config.raster.resolution_m
    crs = config.project.crs_metric
    transform = from_origin(661_000.0, 7_181_000.0, resolution, resolution)

    wc = tmp_path / "interim" / "worldcover.tif"
    write_worldcover(wc, worldcover_array, transform, crs)

    roads_path = tmp_path / "interim" / "osm_roads.gpkg"
    gpd.GeoDataFrame(
        {"highway": ["residential"] * len(lines), "buffer_m": [buffer_m] * len(lines)},
        geometry=lines,
        crs=crs,
    ).to_file(roads_path, driver="GPKG", layer="roads")

    out = tmp_path / "processed" / "impervious_mask.tif"
    monkey = {
        "worldcover": lambda _c: wc,
        "roads": lambda _c: (roads_path, "roads"),
        "impervious_mask": lambda _c: out,
    }
    return artifacts, monkey, out, transform


def test_build_writes_a_binary_mask_on_the_reference_grid(monkeypatch, tmp_path, config):
    worldcover = np.full((50, 50), TREES, dtype="uint8")
    worldcover[0:10, 0:10] = BUILTUP
    line = LineString([(661_050, 7_180_800), (661_450, 7_180_800)])

    artifacts, patches, out, transform = prepare(tmp_path, config, worldcover, [line])
    for name, fn in patches.items():
        monkeypatch.setattr(artifacts, name, fn)

    written = build(config)
    assert written == out

    with rasterio.open(written) as src:
        data = src.read(1)
        assert src.transform == transform
        assert (src.height, src.width) == worldcover.shape
        assert src.crs.to_string() == config.project.crs_metric
        assert set(np.unique(data).tolist()) <= {0, 1}
        # O quadrado construído e a via precisam estar marcados.
        assert data[0:10, 0:10].all()
        assert data.sum() > 100


def test_build_records_the_source_breakdown_in_the_tags(monkeypatch, tmp_path, config):
    worldcover = np.full((40, 40), TREES, dtype="uint8")
    worldcover[0:8, 0:8] = BUILTUP
    line = LineString([(661_050, 7_180_700), (661_350, 7_180_700)])

    artifacts, patches, _out, _ = prepare(tmp_path, config, worldcover, [line])
    for name, fn in patches.items():
        monkeypatch.setattr(artifacts, name, fn)

    with rasterio.open(build(config)) as src:
        tags = src.tags()
        assert "OR" in tags["definition"]
        assert tags["all_touched"] == "False"
        # A via caiu sobre área de árvore, então precisa aparecer como ganho do OSM.
        assert float(tags["pct_osm_only_pct"]) > 0


def test_build_fails_when_an_input_is_missing(monkeypatch, tmp_path, config):
    from floodrisk import artifacts

    monkeypatch.setattr(artifacts, "worldcover", lambda _c: tmp_path / "nao_existe.tif")
    monkeypatch.setattr(artifacts, "roads", lambda _c: (tmp_path / "nada.gpkg", "roads"))
    with pytest.raises(MaskError, match="insumo ausente"):
        build(config)


def test_wider_buffer_produces_more_impervious(monkeypatch, tmp_path, config):
    """Sanidade do buffer hierárquico: largura maior tem que marcar mais pixel."""
    worldcover = np.full((60, 60), TREES, dtype="uint8")
    line = LineString([(661_050, 7_180_700), (661_550, 7_180_700)])

    totals = []
    for index, width in enumerate((4.0, 12.0)):
        folder = tmp_path / f"run{index}"
        artifacts, patches, _out, _ = prepare(folder, config, worldcover, [line], width)
        for name, fn in patches.items():
            monkeypatch.setattr(artifacts, name, fn)
        with rasterio.open(build(config)) as src:
            totals.append(int(src.read(1).sum()))

    assert totals[1] > totals[0]

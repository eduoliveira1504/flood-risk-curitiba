from __future__ import annotations

import json

import pytest
import requests
from shapely.geometry import box, mapping

from floodrisk.acquisition.boundary import (
    BoundaryError,
    acquire,
    fetch_boundary,
    validate_area,
)
from floodrisk.config import load_config


@pytest.fixture
def config():
    return load_config()


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is _BAD_JSON:
            raise ValueError("nope")
        return self._payload


_BAD_JSON = object()


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, timeout=None):
        self.calls.append({"url": url, "timeout": timeout})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def feature_collection(geometry):
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(geometry)}],
    }


def curitiba_like(area_km2: float, config):
    """Polígono em WGS84 cuja área métrica é aproximadamente a pedida."""
    import geopandas as gpd

    side_m = (area_km2 * 1_000_000) ** 0.5
    # Constrói em métrico e volta para geográfico, para a área sair certa.
    centre = (661_000 + side_m / 2, 7_180_000 + side_m / 2)
    square = box(
        centre[0] - side_m / 2,
        centre[1] - side_m / 2,
        centre[0] + side_m / 2,
        centre[1] + side_m / 2,
    )
    series = gpd.GeoSeries([square], crs=config.project.crs_metric).to_crs(
        config.boundary.source_crs
    )
    return series.iloc[0]


# --------------------------------------------------------------------------- #
# Validação de área — a rede de segurança do estágio
# --------------------------------------------------------------------------- #


def test_official_area_passes(config):
    validate_area(435.0, config.boundary)


def test_area_within_tolerance_passes(config):
    low, high = config.boundary.area_range_km2
    validate_area(low + 0.1, config.boundary)
    validate_area(high - 0.1, config.boundary)


def test_wrong_municipality_is_caught(config):
    """São Paulo tem ~1.521 km²; pedir o código errado precisa explodir."""
    with pytest.raises(BoundaryError, match="fora da faixa"):
        validate_area(1521.0, config.boundary)


def test_partial_response_is_caught(config):
    with pytest.raises(BoundaryError, match="fora da faixa"):
        validate_area(12.0, config.boundary)


def test_bbox_area_would_be_rejected(config):
    """A bbox tem ~737 km² — se ela vazasse para cá, o teste pega."""
    with pytest.raises(BoundaryError, match="fora da faixa"):
        validate_area(736.6, config.boundary)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def test_fetch_returns_the_payload(config):
    payload = {"type": "FeatureCollection", "features": []}
    session = FakeSession([FakeResponse(200, payload)])
    assert fetch_boundary(config.boundary, session, sleep=lambda _: None) == payload


def test_fetch_retries_transient_failures(config):
    payload = {"type": "FeatureCollection", "features": []}
    session = FakeSession(
        [requests.ConnectionError("boom"), FakeResponse(503, {}), FakeResponse(200, payload)]
    )
    boundary = config.boundary
    object.__setattr__(boundary, "max_retries", 3)
    assert fetch_boundary(boundary, session, sleep=lambda _: None) == payload


def test_fetch_does_not_retry_404_and_names_the_code(config):
    session = FakeSession([FakeResponse(404, {})])
    with pytest.raises(BoundaryError, match="4106902"):
        fetch_boundary(config.boundary, session, sleep=lambda _: None)
    assert len(session.calls) == 1


def test_fetch_rejects_non_json(config):
    session = FakeSession([FakeResponse(200, _BAD_JSON)])
    with pytest.raises(BoundaryError, match="JSON"):
        fetch_boundary(config.boundary, session, sleep=lambda _: None)


def test_fetch_rejects_a_json_list(config):
    session = FakeSession([FakeResponse(200, [1, 2, 3])])
    with pytest.raises(BoundaryError, match="objeto GeoJSON"):
        fetch_boundary(config.boundary, session, sleep=lambda _: None)


# --------------------------------------------------------------------------- #
# Estágio completo, com a rede fingida
# --------------------------------------------------------------------------- #


def test_acquire_writes_a_valid_boundary(monkeypatch, tmp_path, config):
    import geopandas as gpd

    from floodrisk.acquisition import boundary as module

    target = tmp_path / "limites" / "curitiba.geojson"
    object.__setattr__(config.aoi, "boundary_file", str(target))

    payload = feature_collection(curitiba_like(435.0, config))
    monkeypatch.setattr(module, "fetch_boundary", lambda *a, **k: payload)

    written = acquire(config)
    assert written.exists()

    frame = gpd.read_file(written)
    assert len(frame) == 1
    assert frame.crs.to_string() == config.project.crs_geo
    assert frame["municipality_code"].iloc[0] == config.boundary.municipality_code
    saved = json.loads(written.read_text(encoding="utf-8"))
    assert saved["type"] == "FeatureCollection"


def test_acquire_refuses_a_polygon_of_the_wrong_size(monkeypatch, tmp_path, config):
    from floodrisk.acquisition import boundary as module

    target = tmp_path / "limites" / "errado.geojson"
    object.__setattr__(config.aoi, "boundary_file", str(target))

    payload = feature_collection(curitiba_like(1500.0, config))
    monkeypatch.setattr(module, "fetch_boundary", lambda *a, **k: payload)

    with pytest.raises(BoundaryError, match="fora da faixa"):
        acquire(config)
    assert not target.exists(), "nada pode ser gravado quando a validação falha"


def test_acquire_rejects_non_polygon_geometry(monkeypatch, tmp_path, config):
    from shapely.geometry import Point

    from floodrisk.acquisition import boundary as module

    target = tmp_path / "limites" / "ponto.geojson"
    object.__setattr__(config.aoi, "boundary_file", str(target))
    monkeypatch.setattr(
        module, "fetch_boundary", lambda *a, **k: feature_collection(Point(0, 0))
    )

    with pytest.raises(BoundaryError, match="polígono"):
        acquire(config)


def test_acquire_rejects_an_empty_collection(monkeypatch, tmp_path, config):
    from floodrisk.acquisition import boundary as module

    target = tmp_path / "limites" / "vazio.geojson"
    object.__setattr__(config.aoi, "boundary_file", str(target))
    monkeypatch.setattr(
        module,
        "fetch_boundary",
        lambda *a, **k: {"type": "FeatureCollection", "features": []},
    )

    with pytest.raises(BoundaryError, match="nenhuma feição"):
        acquire(config)


def test_acquire_is_idempotent_and_does_not_refetch(monkeypatch, tmp_path, config):
    from floodrisk.acquisition import boundary as module

    target = tmp_path / "limites" / "ja_existe.geojson"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    object.__setattr__(config.aoi, "boundary_file", str(target))

    def explode(*args, **kwargs):
        raise AssertionError("não deveria baixar com o arquivo já presente")

    monkeypatch.setattr(module, "fetch_boundary", explode)
    assert acquire(config) == target.resolve()

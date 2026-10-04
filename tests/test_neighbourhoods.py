from __future__ import annotations

import pytest

from floodrisk.acquisition.neighbourhoods import (
    NeighbourhoodsError,
    display_name,
    parse_features,
    polygon_from_rings,
)
from floodrisk.config import load_config
from floodrisk.pipeline import STAGES_BY_NAME
from floodrisk.report import index_neighbourhoods

# Quadrado de 100 m em sentido HORÁRIO — contorno externo na convenção da Esri.
OUTER = [[0, 0], [0, 100], [100, 100], [100, 0], [0, 0]]
# Quadrado de 20 m em sentido ANTI-HORÁRIO — furo.
HOLE = [[40, 40], [60, 40], [60, 60], [40, 60], [40, 40]]


@pytest.fixture
def settings():
    return load_config().neighbourhoods


def feature(name="CAMPO DE SANTANA", rings=None):
    return {
        "attributes": {"objectid": 1, "nome": name, "nm_regional": "REGIONAL TATUQUARA"},
        "geometry": {"rings": rings or [OUTER]},
    }


def test_names_are_written_as_people_write_them():
    assert display_name("CAMPO DE SANTANA") == "Campo de Santana"
    assert display_name("ALTO DA XV") == "Alto da XV"
    assert display_name("  CENTRO ") == "Centro"


def test_a_leading_particle_is_still_capitalised():
    assert display_name("DAS FLORES") == "Das Flores"


def test_a_blank_name_becomes_missing():
    assert display_name("  ") is None
    assert display_name(None) is None


def test_a_clockwise_ring_is_the_outline():
    assert polygon_from_rings([OUTER]).area == pytest.approx(10_000)


def test_a_counter_clockwise_ring_is_a_hole():
    assert polygon_from_rings([OUTER, HOLE]).area == pytest.approx(10_000 - 400)


def test_rings_outside_the_convention_are_kept_as_outlines():
    """Melhor um bairro sem furo do que um bairro perdido."""
    assert polygon_from_rings([OUTER[::-1]]).area == pytest.approx(10_000)


def test_a_degenerate_ring_yields_nothing():
    assert polygon_from_rings([[[0, 0], [1, 1]]]) is None


def test_a_feature_becomes_a_named_polygon(settings):
    records = parse_features({"features": [feature()]}, settings)
    assert records[0]["name"] == "Campo de Santana"
    assert records[0]["region"] == "Regional Tatuquara"
    assert records[0]["geometry"].area == pytest.approx(10_000)


def test_a_feature_without_a_name_is_skipped(settings):
    assert parse_features({"features": [feature(name=None)]}, settings) == []


def test_a_service_error_is_raised(settings):
    with pytest.raises(NeighbourhoodsError, match="recusou"):
        parse_features({"error": {"code": 503}}, settings)


def test_the_stage_is_registered():
    assert STAGES_BY_NAME["acquire-neighbourhoods"].implemented


def test_neighbourhood_names_become_indices():
    data = {
        "type": "FeatureCollection",
        "features": [
            {"properties": {"neighbourhood": "Centro"}},
            {"properties": {"neighbourhood": None}},
            {"properties": {"neighbourhood": "Batel"}},
        ],
    }
    assert index_neighbourhoods(data) == ["Batel", "Centro"]
    assert [f["properties"]["neighbourhood_index"] for f in data["features"]] == [1, -1, 0]

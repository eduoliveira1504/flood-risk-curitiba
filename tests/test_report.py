from __future__ import annotations

import json

import pytest

from floodrisk.report import (
    COORDINATE_DECIMALS,
    WEB_PROPERTIES,
    ReportError,
    compact_geojson,
    round_coordinates,
)


def cell(x=-49.2733123456789, y=-25.4284987654321, **properties):
    ring = [[x, y], [x + 0.002, y], [x + 0.002, y + 0.002], [x, y + 0.002], [x, y]]
    base = {
        "cell_id": 7,
        "class_index": 4,
        "class_label": "muito alta",
        "susceptibility": 0.812345678,
        "impervious_mean": 0.9551234,
        "slope_mean_deg": 2.4123456,
        "flatness": 0.77123,
        "pixels": 400,
    }
    base.update(properties)
    return {
        "type": "Feature",
        "properties": base,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def collection(*features):
    return {"type": "FeatureCollection", "features": list(features)}


# --------------------------------------------------------------------------- #
# Arredondamento de coordenadas
# --------------------------------------------------------------------------- #


def test_a_bare_number_is_rounded():
    assert round_coordinates(-49.2733123456789, 5) == -49.27331


def test_nesting_depth_does_not_matter():
    """Point, Polygon e MultiPolygon aninham a profundidades diferentes."""
    deep = [[[[-49.123456789, -25.987654321]]]]
    assert round_coordinates(deep, 3) == [[[[-49.123, -25.988]]]]


def test_non_numeric_values_pass_through():
    assert round_coordinates("Polygon", 5) == "Polygon"
    assert round_coordinates(None, 5) is None


def test_booleans_are_not_treated_as_numbers():
    """bool é subclasse de int em Python; arredondá-lo viraria 1.0 e quebraria o JSON."""
    assert round_coordinates(True, 5) is True


def test_five_decimals_keep_metre_precision():
    """1e-5 grau ≈ 1,1 m — de sobra para célula de 200 m."""
    original = -25.428498
    rounded = round_coordinates(original, COORDINATE_DECIMALS)
    assert abs(rounded - original) * 111_000 < 1.5


# --------------------------------------------------------------------------- #
# Compactação
# --------------------------------------------------------------------------- #


def test_only_the_web_properties_survive():
    out = compact_geojson(collection(cell()))
    assert set(out["features"][0]["properties"]) == set(WEB_PROPERTIES.values())


def test_the_properties_travel_under_short_names():
    """Os nomes longos custam centenas de kB repetidos 11.239 vezes."""
    properties = compact_geojson(collection(cell()))["features"][0]["properties"]
    assert "impervious_mean" not in properties
    assert properties["i"] == pytest.approx(0.955, abs=1e-9)


def test_the_short_names_are_unique():
    """Duas propriedades com a mesma letra se sobrescreveriam em silêncio."""
    assert len(set(WEB_PROPERTIES.values())) == len(WEB_PROPERTIES)


def test_the_geometry_type_is_preserved():
    out = compact_geojson(collection(cell()))
    assert out["features"][0]["geometry"]["type"] == "Polygon"


def test_the_ring_still_closes():
    """Anel que não fecha é polígono inválido — o arredondamento não pode abri-lo."""
    ring = compact_geojson(collection(cell()))["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]


def test_every_feature_is_kept():
    out = compact_geojson(collection(cell(), cell(x=-49.2), cell(x=-49.1)))
    assert len(out["features"]) == 3


def test_features_without_geometry_are_dropped():
    broken = {"type": "Feature", "properties": {"class_index": 1}, "geometry": None}
    assert len(compact_geojson(collection(cell(), broken))["features"]) == 1


def test_integer_properties_stay_integers():
    """class_index indexa um array de cores no JS; virar float quebraria."""
    value = compact_geojson(collection(cell()))["features"][0]["properties"]["c"]
    assert isinstance(value, int)


def test_each_property_keeps_its_declared_precision():
    """Guardar 4 casas de uma declividade exibida como "2,4°" é pagar transporte
    por dígito que ninguém vê."""
    out = compact_geojson(collection(cell()))["features"][0]["properties"]
    assert out["s"] == pytest.approx(0.812, abs=1e-9)
    assert out["d"] == pytest.approx(2.41, abs=1e-9)


def test_a_missing_property_does_not_crash():
    feature = cell()
    del feature["properties"]["slope_mean_deg"]
    out = compact_geojson(collection(feature))["features"][0]["properties"]
    assert "d" not in out
    assert "c" in out


def test_compaction_actually_shrinks_the_payload():
    """É a razão de existir do estágio: 5,6 MB não podem ir para uma página web."""
    original = collection(*[cell(x=-49.3 + 0.002 * i) for i in range(200)])
    before = len(json.dumps(original))
    after = len(json.dumps(compact_geojson(original), separators=(",", ":")))
    assert after < before * 0.6


def test_a_non_collection_is_rejected():
    with pytest.raises(ReportError, match="FeatureCollection"):
        compact_geojson({"type": "Feature", "properties": {}, "geometry": None})


def test_a_collection_without_features_is_rejected():
    with pytest.raises(ReportError, match="lista de feições"):
        compact_geojson({"type": "FeatureCollection"})


def test_a_collection_where_nothing_survives_is_rejected():
    broken = {"type": "Feature", "properties": {}, "geometry": None}
    with pytest.raises(ReportError, match="nenhuma feição"):
        compact_geojson(collection(broken))

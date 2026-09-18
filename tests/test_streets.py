from __future__ import annotations

import dataclasses

import pytest
import requests
from shapely.geometry import LineString, MultiLineString

from floodrisk.acquisition.streets import (
    StreetsError,
    buffer_for,
    fetch_all,
    page_params,
    parse_features,
)
from floodrisk.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def streets(config):
    return config.ground_truth.streets


# --------------------------------------------------------------------------- #
# Parâmetros da consulta
# --------------------------------------------------------------------------- #


def test_params_ask_for_esri_json_not_geojson(streets):
    """GeoJSON obriga WGS84; pedir GeoJSON com outSR gera arquivo que mente."""
    assert page_params(streets, 0)["f"] == "json"


def test_params_request_the_analysis_crs(streets):
    assert page_params(streets, 0)["outSR"] == "31982"


def test_params_always_order_for_stable_pagination(streets):
    """Sem ordem estável, resultOffset repete e pula feições entre páginas."""
    assert page_params(streets, 0)["orderByFields"] == streets.order_by_field


def test_params_carry_the_offset_and_page_size(streets):
    params = page_params(streets, 4000)
    assert params["resultOffset"] == 4000
    assert params["resultRecordCount"] == streets.page_size


def test_params_include_the_hierarchy_field(streets):
    assert streets.hierarchy_field in page_params(streets, 0)["outFields"]


def test_negative_offset_is_rejected(streets):
    with pytest.raises(StreetsError, match="offset"):
        page_params(streets, -1)


# --------------------------------------------------------------------------- #
# Buffer por hierarquia
# --------------------------------------------------------------------------- #


def test_known_hierarchy_uses_its_own_buffer():
    assert buffer_for("COLETORA 1", {"COLETORA 1": 8.0}, 5.0) == 8.0


def test_matching_ignores_case_and_surrounding_space():
    """O cadastro mistura 'COLETORA 1' e 'Coletora 1' entre camadas."""
    buffers = {"Coletora 1": 8.0}
    assert buffer_for("  coletora 1 ", buffers, 5.0) == 8.0
    assert buffer_for("COLETORA 1", buffers, 5.0) == 8.0


def test_unknown_hierarchy_falls_back_to_the_default():
    """Valor novo no cadastro não pode derrubar o estágio."""
    assert buffer_for("VIA NOVA", {"COLETORA 1": 8.0}, 5.0) == 5.0


def test_missing_hierarchy_falls_back_to_the_default():
    assert buffer_for(None, {"COLETORA 1": 8.0}, 5.0) == 5.0


# --------------------------------------------------------------------------- #
# Interpretação da resposta Esri
# --------------------------------------------------------------------------- #


def feature(object_id, hierarchy, paths, name="Rua Teste"):
    return {
        "attributes": {
            "objectid": object_id,
            "gtm_nm_logradouro": name,
            "hierarquia_viaria": hierarchy,
        },
        "geometry": {"paths": paths},
    }


SEGMENT = [[661_000.0, 7_181_000.0], [661_100.0, 7_181_000.0]]


def test_single_path_becomes_a_linestring(streets):
    records = parse_features({"features": [feature(1, "COLETORA 1", [SEGMENT])]}, streets)
    assert isinstance(records[0]["geometry"], LineString)
    assert records[0]["objectid"] == 1
    assert records[0]["hierarchy"] == "COLETORA 1"


def test_multiple_paths_become_a_multilinestring(streets):
    other = [[662_000.0, 7_182_000.0], [662_100.0, 7_182_000.0]]
    records = parse_features({"features": [feature(2, None, [SEGMENT, other])]}, streets)
    assert isinstance(records[0]["geometry"], MultiLineString)


def test_coordinates_keep_the_metric_values(streets):
    records = parse_features({"features": [feature(1, None, [SEGMENT])]}, streets)
    assert list(records[0]["geometry"].coords) == [
        (661_000.0, 7_181_000.0),
        (661_100.0, 7_181_000.0),
    ]


def test_extra_ordinates_are_ignored(streets):
    """O serviço pode devolver M ou Z junto de X e Y."""
    path = [[661_000.0, 7_181_000.0, 5.0], [661_100.0, 7_181_000.0, 6.0]]
    records = parse_features({"features": [feature(1, None, [path])]}, streets)
    assert list(records[0]["geometry"].coords) == [
        (661_000.0, 7_181_000.0),
        (661_100.0, 7_181_000.0),
    ]


def test_features_without_geometry_are_skipped(streets):
    payload = {
        "features": [
            feature(1, None, [SEGMENT]),
            {"attributes": {"objectid": 2}, "geometry": {}},
            {"attributes": {"objectid": 3}, "geometry": {"paths": []}},
        ]
    }
    assert len(parse_features(payload, streets)) == 1


def test_single_point_path_is_skipped(streets):
    payload = {"features": [feature(1, None, [[[661_000.0, 7_181_000.0]]])]}
    assert parse_features(payload, streets) == []


def test_service_error_is_surfaced(streets):
    payload = {"error": {"code": 400, "message": "Invalid layer"}}
    with pytest.raises(StreetsError, match="recusou a consulta"):
        parse_features(payload, streets)


def test_missing_features_list_is_rejected(streets):
    with pytest.raises(StreetsError, match="features"):
        parse_features({}, streets)


# --------------------------------------------------------------------------- #
# Paginação
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def page(count, exceeded, start=0):
    return {
        "features": [
            feature(start + i, "COLETORA 1", [SEGMENT]) for i in range(count)
        ],
        "exceededTransferLimit": exceeded,
    }


def test_pagination_walks_until_the_service_stops(streets):
    streets = dataclasses.replace(streets, page_size=3)
    session = FakeSession(
        [
            FakeResponse(200, page(3, True, 0)),
            FakeResponse(200, page(3, True, 3)),
            FakeResponse(200, page(1, False, 6)),
        ]
    )
    records = fetch_all(streets, session, sleep=lambda _: None)
    assert len(records) == 7
    assert [c["params"]["resultOffset"] for c in session.calls] == [0, 3, 6]


def test_pagination_stops_on_a_short_page(streets):
    streets = dataclasses.replace(streets, page_size=10)
    session = FakeSession([FakeResponse(200, page(4, False))])
    assert len(fetch_all(streets, session, sleep=lambda _: None)) == 4
    assert len(session.calls) == 1


def test_pagination_guards_against_an_endless_loop(streets):
    """Serviço que sempre diz 'tem mais' não pode travar o estágio para sempre."""
    streets = dataclasses.replace(streets, page_size=2, max_pages=3)
    session = FakeSession([FakeResponse(200, page(2, True))] * 10)
    with pytest.raises(StreetsError, match="laço infinito"):
        fetch_all(streets, session, sleep=lambda _: None)


def test_empty_result_is_rejected(streets):
    session = FakeSession([FakeResponse(200, {"features": [], "exceededTransferLimit": False})])
    with pytest.raises(StreetsError, match="nenhuma via"):
        fetch_all(streets, session, sleep=lambda _: None)


def test_transient_failure_is_retried(streets):
    streets = dataclasses.replace(streets, page_size=10, max_retries=3)
    session = FakeSession(
        [
            requests.ConnectionError("boom"),
            FakeResponse(503, {}),
            FakeResponse(200, page(2, False)),
        ]
    )
    assert len(fetch_all(streets, session, sleep=lambda _: None)) == 2


def test_client_error_is_not_retried(streets):
    session = FakeSession([FakeResponse(404, {})])
    with pytest.raises(StreetsError, match="layer_id"):
        fetch_all(streets, session, sleep=lambda _: None)
    assert len(session.calls) == 1


# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #


def raw_config():
    import yaml

    from floodrisk.config import find_repo_root

    with (find_repo_root() / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build(data):
    from floodrisk.config import Config, find_repo_root

    return Config.from_dict(data, root=find_repo_root())


def test_hierarchy_field_must_be_downloaded():
    """Pedir buffer por um campo que não está em out_fields daria sempre o padrão."""
    import copy

    from floodrisk.config import ConfigError

    data = copy.deepcopy(raw_config())
    data["ground_truth"]["streets"]["hierarchy_field"] = "campo_inexistente"
    with pytest.raises(ConfigError, match="out_fields"):
        build(data)


def test_order_field_must_be_downloaded():
    import copy

    from floodrisk.config import ConfigError

    data = copy.deepcopy(raw_config())
    data["ground_truth"]["streets"]["order_by_field"] = "nao_pedido"
    with pytest.raises(ConfigError, match="paginação"):
        build(data)


def test_page_size_above_the_service_limit_is_rejected():
    """O serviço ignora page_size > 2000 em silêncio e a paginação pula feições."""
    import copy

    from floodrisk.config import ConfigError

    data = copy.deepcopy(raw_config())
    data["ground_truth"]["streets"]["page_size"] = 5000
    with pytest.raises(ConfigError, match="page_size"):
        build(data)


def test_unknown_roads_source_is_rejected():
    import copy

    from floodrisk.config import ConfigError

    data = copy.deepcopy(raw_config())
    data["ground_truth"]["roads_source"] = "google"
    with pytest.raises(ConfigError, match="roads_source"):
        build(data)


def test_roads_artifact_follows_the_configured_source():
    import copy

    from floodrisk import artifacts

    data = copy.deepcopy(raw_config())
    data["ground_truth"]["roads_source"] = "geocuritiba"
    path, layer = artifacts.roads(build(data))
    assert path.name == "streets.gpkg" and layer == "streets"

    data["ground_truth"]["roads_source"] = "osm"
    path, layer = artifacts.roads(build(data))
    assert path.name == "osm_roads.gpkg" and layer == "roads"

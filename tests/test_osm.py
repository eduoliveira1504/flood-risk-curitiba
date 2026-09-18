from __future__ import annotations

import re

import pytest
import requests

from floodrisk.acquisition.osm import (
    OSMError,
    build_query,
    fetch,
    parse_ways,
)
from floodrisk.config import load_config

CWB_BBOX = (-49.3960, -25.6560, -49.1840, -25.3440)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def buffers(config):
    return config.ground_truth.osm.road_buffers_m


# --------------------------------------------------------------------------- #
# Consulta
# --------------------------------------------------------------------------- #


def test_query_uses_overpass_bbox_order():
    """Overpass usa (sul, oeste, norte, leste) — o inverso do resto do mundo geo.

    Trocar a ordem devolve área vazia ou erro, nunca um resultado obviamente
    errado. É o tipo de bug que se descobre tarde.
    """
    query = build_query(CWB_BBOX, ["residential"], 300)
    west, south, east, north = CWB_BBOX
    assert f"({south},{west},{north},{east})" in query


def test_query_filters_exactly_the_configured_types(buffers):
    query = build_query(CWB_BBOX, list(buffers), 300)
    listed = re.search(r'highway"~"\^\(([^)]*)\)', query).group(1).split("|")
    assert set(listed) == set(buffers)


def test_query_is_stable_across_calls(buffers):
    """Consulta estável é cache estável — a ordem dos tipos não pode variar."""
    a = build_query(CWB_BBOX, list(buffers), 300)
    b = build_query(CWB_BBOX, list(reversed(list(buffers))), 300)
    assert a == b


def test_query_carries_the_timeout():
    assert "[out:json][timeout:180];" in build_query(CWB_BBOX, ["primary"], 180)


def test_query_requests_inline_geometry():
    """Sem 'out geom' seria preciso resolver referências de nó à mão."""
    assert build_query(CWB_BBOX, ["primary"], 300).rstrip().endswith("out geom;")


def test_query_rejects_empty_types():
    with pytest.raises(OSMError, match="nenhum tipo"):
        build_query(CWB_BBOX, [], 300)


def test_query_rejects_degenerate_bbox():
    with pytest.raises(OSMError, match="degenerado"):
        build_query((0, 0, 0, 0), ["primary"], 300)


# --------------------------------------------------------------------------- #
# Interpretação da resposta
# --------------------------------------------------------------------------- #


def way(osm_id, highway, points):
    return {
        "type": "way",
        "id": osm_id,
        "tags": {"highway": highway},
        "geometry": [{"lat": lat, "lon": lon} for lon, lat in points],
    }


LINE = [(-49.27, -25.43), (-49.26, -25.42)]


def test_parse_attaches_the_buffer_of_each_type(buffers):
    payload = {"elements": [way(1, "motorway", LINE), way(2, "service", LINE)]}
    records = parse_ways(payload, buffers)
    widths = {r["highway"]: r["buffer_m"] for r in records}
    assert widths == {"motorway": buffers["motorway"], "service": buffers["service"]}


def test_parse_builds_linestrings_in_lon_lat_order(buffers):
    records = parse_ways({"elements": [way(1, "primary", LINE)]}, buffers)
    assert list(records[0]["geometry"].coords) == [(-49.27, -25.43), (-49.26, -25.42)]


def test_parse_drops_ways_with_a_single_point(buffers):
    """Um ponto não faz linha; o shapely só reclamaria na hora do buffer."""
    payload = {
        "elements": [
            way(1, "primary", LINE),
            way(2, "primary", [(-49.27, -25.43)]),
        ]
    }
    assert len(parse_ways(payload, buffers)) == 1


def test_parse_ignores_types_outside_the_configuration(buffers):
    payload = {"elements": [way(1, "primary", LINE), way(2, "footway", LINE)]}
    records = parse_ways(payload, buffers)
    assert [r["highway"] for r in records] == ["primary"]


def test_parse_ignores_non_way_elements(buffers):
    payload = {
        "elements": [
            {"type": "node", "id": 9, "lat": -25.4, "lon": -49.2},
            way(1, "primary", LINE),
        ]
    }
    assert len(parse_ways(payload, buffers)) == 1


def test_parse_rejects_a_payload_without_elements(buffers):
    with pytest.raises(OSMError, match="elements"):
        parse_ways({}, buffers)


def test_parse_rejects_an_empty_result(buffers):
    with pytest.raises(OSMError, match="nenhuma via utilizável"):
        parse_ways({"elements": []}, buffers)


def test_parse_reports_malformed_geometry(buffers):
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 7,
                "tags": {"highway": "primary"},
                "geometry": [{"lat": -25.4}, {"lat": -25.3, "lon": -49.2}],
            }
        ]
    }
    with pytest.raises(OSMError, match="malformada"):
        parse_ways(payload, buffers)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is _BAD:
            raise ValueError("nope")
        return self._payload


_BAD = object()


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.headers = {}

    def post(self, url, data=None, timeout=None):
        self.calls.append({"url": url, "data": data, "timeout": timeout})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_fetch_posts_the_query_and_returns_the_payload(config):
    payload = {"elements": []}
    session = FakeSession([FakeResponse(200, payload)])
    assert fetch("QUERY", config.ground_truth.osm, session, sleep=lambda _: None) == payload
    assert session.calls[0]["data"] == {"data": "QUERY"}
    assert session.calls[0]["url"] == config.ground_truth.osm.overpass_urls[0]


def test_406_moves_to_the_next_mirror_without_retrying(config):
    """O overpass-api.de recusa cliente programático com 406, de forma persistente.

    Insistir no mesmo host é inútil; o certo é trocar de espelho na hora.
    """
    session = FakeSession([FakeResponse(406, {}), FakeResponse(200, {"elements": []})])
    fetch("Q", config.ground_truth.osm, session, sleep=lambda _: None)

    assert len(session.calls) == 2, "406 não pode ser repetido no mesmo espelho"
    urls = [c["url"] for c in session.calls]
    assert urls[0] != urls[1]
    assert urls == config.ground_truth.osm.overpass_urls[:2]


def test_403_also_moves_on(config):
    session = FakeSession([FakeResponse(403, {}), FakeResponse(200, {"elements": []})])
    fetch("Q", config.ground_truth.osm, session, sleep=lambda _: None)
    assert len(session.calls) == 2


def test_every_mirror_refusing_names_them_all(config):
    osm = config.ground_truth.osm
    session = FakeSession([FakeResponse(406, {})] * len(osm.overpass_urls))
    with pytest.raises(OSMError, match="todos os espelhos"):
        fetch("Q", osm, session, sleep=lambda _: None)
    assert len(session.calls) == len(osm.overpass_urls)


def test_exhausted_retries_fall_through_to_the_next_mirror(config):
    """Espelho que só devolve 429 é abandonado depois das tentativas."""
    osm = config.ground_truth.osm
    responses = [FakeResponse(429, {})] * osm.max_retries + [
        FakeResponse(200, {"elements": []})
    ]
    session = FakeSession(responses)
    fetch("Q", osm, session, sleep=lambda _: None)

    urls = [c["url"] for c in session.calls]
    assert urls[: osm.max_retries] == [osm.overpass_urls[0]] * osm.max_retries
    assert urls[-1] == osm.overpass_urls[1]


def test_user_agent_is_ascii_only():
    """Cabeçalho HTTP com acento é recusa na certa em alguns servidores."""
    from floodrisk.acquisition.osm import _HEADERS

    for name, value in _HEADERS.items():
        value.encode("ascii"), name.encode("ascii")


def test_fetch_retries_on_rate_limit(config):
    """429 é rotina no Overpass público — desistir na primeira seria frágil."""
    session = FakeSession([FakeResponse(429, {}), FakeResponse(200, {"elements": []})])
    assert fetch("Q", config.ground_truth.osm, session, sleep=lambda _: None) == {
        "elements": []
    }
    # Repetido no MESMO espelho, porque 429 é transitório.
    assert session.calls[0]["url"] == session.calls[1]["url"]


def test_fetch_retries_on_gateway_timeout(config):
    session = FakeSession([FakeResponse(504, {}), FakeResponse(200, {"elements": []})])
    fetch("Q", config.ground_truth.osm, session, sleep=lambda _: None)


def test_fetch_retries_on_network_error(config):
    session = FakeSession(
        [requests.ConnectionError("boom"), FakeResponse(200, {"elements": []})]
    )
    fetch("Q", config.ground_truth.osm, session, sleep=lambda _: None)


def test_a_bad_query_fails_immediately_without_trying_other_mirrors(config):
    """400 é erro nosso: os outros espelhos dariam o mesmo. Falhar já."""
    session = FakeSession([FakeResponse(400, {})])
    with pytest.raises(OSMError, match="consulta inválida"):
        fetch("Q", config.ground_truth.osm, session, sleep=lambda _: None)
    assert len(session.calls) == 1


def test_fetch_gives_up_when_no_mirror_answers(config):
    osm = config.ground_truth.osm
    session = FakeSession(
        [FakeResponse(429, {})] * osm.max_retries * len(osm.overpass_urls)
    )
    with pytest.raises(OSMError, match="todos os espelhos"):
        fetch("Q", osm, session, sleep=lambda _: None)


def test_fetch_rejects_non_json(config):
    session = FakeSession([FakeResponse(200, _BAD)])
    with pytest.raises(OSMError, match="JSON"):
        fetch("Q", config.ground_truth.osm, session, sleep=lambda _: None)


# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #


def test_non_positive_buffer_is_rejected():
    import copy

    import yaml

    from floodrisk.config import Config, ConfigError, find_repo_root

    with (find_repo_root() / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    data = copy.deepcopy(raw)
    data["ground_truth"]["osm"]["road_buffers_m"]["primary"] = 0
    with pytest.raises(ConfigError, match="buffer não positivo"):
        Config.from_dict(data, root=find_repo_root())


def test_empty_mirror_list_is_rejected():
    import copy

    import yaml

    from floodrisk.config import Config, ConfigError, find_repo_root

    with (find_repo_root() / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    data = copy.deepcopy(raw)
    data["ground_truth"]["osm"]["overpass_urls"] = []
    with pytest.raises(ConfigError, match="espelho"):
        Config.from_dict(data, root=find_repo_root())

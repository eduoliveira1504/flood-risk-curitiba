"""Malha viária do OpenStreetMap — a outra metade do rótulo da U-Net.

O WorldCover enxerga quadra construída, mas perde rua: a 10 m, via estreita
ladeada de árvore vira pixel misto e o classificador a joga em "árvores". O OSM
tem a geometria exata dessas vias. Juntas, as duas fontes cobrem telhado e
asfalto.

Decisões:

1. **Overpass com ``out geom``.** A geometria vem embutida em cada via, sem
   precisar resolver referências de nó à mão. Custa mais bytes e poupa uma
   classe inteira de bug.
2. **Só os tipos que serão bufferizados.** A consulta filtra exatamente as
   chaves de ``ground_truth.osm.road_buffers_m``. Baixar trilha e ciclovia para
   descartar depois é desperdício de banda e de paciência do Overpass.
3. **A resposta crua fica em ``data/raw/``.** Overpass é serviço comunitário com
   limite de taxa; repetir o estágio lê do disco em vez de bater no servidor de
   novo.
4. **Aqui só se baixa.** O buffer e a rasterização são do ``build-mask``. Este
   estágio entrega vetor em CRS métrico, com a largura de cada via anotada.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import requests

from .. import artifacts
from ..config import Config, OSMConfig

logger = logging.getLogger(__name__)

__all__ = [
    "MirrorRefused",
    "OSMError",
    "acquire",
    "build_query",
    "parse_ways",
]

# Somente ASCII: cabeçalho HTTP com acento é latin-1 no melhor caso e motivo de
# recusa no pior. Identificação descritiva com contato reduz a chance de o
# servidor classificar a chamada como robô.
_USER_AGENT = (
    "flood-risk-curitiba/0.1 (TCC, FAE Centro Universitario, Curitiba BR; academic use)"
)

_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

# 429 = limite de taxa, 5xx = falha do lado deles. Vale repetir no mesmo espelho.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# O servidor recusa ESTE CLIENTE, não esta consulta. O overpass-api.de passou a
# filtrar tráfego programático e devolve 406 de forma persistente — insistir no
# mesmo host não resolve, trocar de espelho resolve.
_MIRROR_REFUSED_STATUS = frozenset({403, 406, 410})

METRES_PER_KM = 1000.0


class OSMError(RuntimeError):
    """Falha ao consultar, interpretar ou validar a resposta do Overpass."""


class MirrorRefused(OSMError):
    """O espelho recusou o cliente. Sinaliza para tentar o próximo, não repetir."""


def build_query(
    bounds: Sequence[float], highway_types: Sequence[str], timeout_s: int
) -> str:
    """Monta a consulta Overpass QL para as vias de interesse.

    Atenção à ordem da bbox: o Overpass usa ``(sul, oeste, norte, leste)``,
    enquanto o resto do mundo geoespacial usa ``(oeste, sul, leste, norte)``.
    Trocar os dois devolve área vazia ou erro, nunca um resultado obviamente
    errado — por isso existe teste para isso.
    """
    if not highway_types:
        raise OSMError("nenhum tipo de via pedido")

    west, south, east, north = bounds
    if west >= east or south >= north:
        raise OSMError(f"envelope degenerado: {tuple(bounds)}")

    # Ordenado para que a consulta seja idêntica entre execuções: consulta
    # estável é cache estável.
    pattern = "|".join(sorted(highway_types))
    return (
        f"[out:json][timeout:{timeout_s}];\n"
        f'way["highway"~"^({pattern})$"]'
        f"({south},{west},{north},{east});\n"
        "out geom;"
    )


def fetch(
    query: str,
    osm: OSMConfig,
    session: requests.Session | None = None,
    sleep=time.sleep,
) -> dict:
    """Executa a consulta, percorrendo os espelhos até um responder.

    Três classes de falha, três reações:

    - **Recusa do espelho** (403/406/410): o servidor não atende este cliente.
      É persistente — o overpass-api.de filtra tráfego programático — então
      repetir é inútil e o próximo espelho é tentado imediatamente.
    - **Transitória** (429/5xx): repete no mesmo espelho com recuo e, se esgotar,
      passa ao próximo.
    - **Consulta inválida** (demais 4xx): erro nosso. Falha na hora, sem tentar
      os outros — todos dariam o mesmo.
    """
    http = session or requests.Session()
    for header, value in _HEADERS.items():
        http.headers.setdefault(header, value)

    refusals: list[str] = []

    for url in osm.overpass_urls:
        try:
            return _fetch_from(url, query, osm, http, sleep)
        except MirrorRefused as exc:
            refusals.append(f"{url}: {exc}")
            logger.warning("espelho recusou (%s), tentando o próximo", exc)

    raise OSMError(
        "todos os espelhos do Overpass recusaram a consulta:\n  "
        + "\n  ".join(refusals)
        + "\nAjuste 'ground_truth.osm.overpass_urls' ou tente mais tarde."
    )


def _fetch_from(
    url: str, query: str, osm: OSMConfig, http, sleep
) -> dict:
    """Consulta um espelho específico, com repetição em falha transitória."""
    last_error: Exception | None = None
    logger.info("consultando %s", url)

    for attempt in range(1, osm.max_retries + 1):
        try:
            response = http.post(
                url, data={"data": query}, timeout=osm.request_timeout_s
            )
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("tentativa %d falhou na rede: %s", attempt, exc)
        else:
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise OSMError(f"resposta do Overpass não é JSON: {exc}") from exc
                if not isinstance(payload, dict):
                    raise OSMError(
                        f"esperado um objeto JSON, veio {type(payload).__name__}"
                    )
                return payload

            if response.status_code in _MIRROR_REFUSED_STATUS:
                raise MirrorRefused(
                    f"HTTP {response.status_code} — o servidor recusa este cliente, "
                    "não a consulta"
                )

            if response.status_code not in _RETRYABLE_STATUS:
                raise OSMError(
                    f"HTTP {response.status_code} em {url}: consulta inválida. "
                    "Trocar de espelho não resolve."
                )

            last_error = OSMError(f"HTTP {response.status_code}")
            logger.warning(
                "tentativa %d recebeu HTTP %d — limite de taxa ou falha do servidor",
                attempt,
                response.status_code,
            )

        if attempt < osm.max_retries:
            # Recuo mais generoso que o habitual: o Overpass é serviço
            # comunitário e insistir rápido só aumenta o bloqueio.
            backoff = 5.0 * attempt
            logger.info("aguardando %.0fs", backoff)
            sleep(backoff)

    raise MirrorRefused(f"falhou após {osm.max_retries} tentativas ({last_error})")


def parse_ways(payload: Mapping, road_buffers_m: Mapping[str, float]) -> list[dict]:
    """Converte a resposta do Overpass em registros com geometria e largura.

    Vias com menos de dois pontos são descartadas: não formam linha, e o
    ``shapely`` levantaria erro só na hora do buffer, muito depois.
    """
    from shapely.geometry import LineString

    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise OSMError("resposta do Overpass sem a lista 'elements'")

    records: list[dict] = []
    skipped_short = 0
    skipped_untyped = 0

    for element in elements:
        if not isinstance(element, dict) or element.get("type") != "way":
            continue

        highway = (element.get("tags") or {}).get("highway")
        if highway not in road_buffers_m:
            skipped_untyped += 1
            continue

        geometry = element.get("geometry")
        if not isinstance(geometry, list) or len(geometry) < 2:
            skipped_short += 1
            continue

        try:
            line = LineString([(point["lon"], point["lat"]) for point in geometry])
        except (KeyError, TypeError) as exc:
            raise OSMError(f"geometria malformada na via {element.get('id')}: {exc}") from exc

        records.append(
            {
                "osm_id": element.get("id"),
                "highway": highway,
                "buffer_m": float(road_buffers_m[highway]),
                "geometry": line,
            }
        )

    if skipped_short:
        logger.warning("%d via(s) com menos de dois pontos, descartadas", skipped_short)
    if skipped_untyped:
        logger.debug("%d elemento(s) de tipo não configurado, ignorados", skipped_untyped)
    if not records:
        raise OSMError(
            "nenhuma via utilizável na resposta — confira a bbox e os tipos "
            "em 'ground_truth.osm.road_buffers_m'"
        )
    return records


def acquire(config: Config) -> Path:
    """Estágio ``acquire-osm``: baixa a malha viária e grava em CRS métrico."""
    import geopandas as gpd

    from ..geo import acquisition_geometry

    osm = config.ground_truth.osm
    aoi = acquisition_geometry(config, metric=False)
    query = build_query(aoi.bounds, list(osm.road_buffers_m), int(osm.request_timeout_s))

    raw_path = config.path("data_raw") / "osm" / "overpass_highways.json"
    if raw_path.exists():
        logger.info(
            "usando resposta em cache: %s (apague para rebaixar)",
            config.display_path(raw_path),
        )
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        logger.debug("consulta:\n%s", query)
        payload = fetch(query, osm)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("resposta crua salva em %s", config.display_path(raw_path))

    records = parse_ways(payload, osm.road_buffers_m)
    frame = gpd.GeoDataFrame(records, crs=config.project.crs_geo).to_crs(
        config.project.crs_metric
    )

    destination = artifacts.osm_roads(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(destination, driver="GPKG", layer="roads")

    _report(frame, config)
    logger.info("malha viária escrita em %s", config.display_path(destination))
    return destination


def _report(frame, config: Config) -> None:
    """Log de extensão por tipo de via — o sinal mais rápido de algo errado."""
    total_km = frame.length.sum() / METRES_PER_KM
    logger.info("vias: %s, extensão total: %.0f km", f"{len(frame):,}", total_km)

    by_type = frame.groupby("highway")["geometry"].apply(
        lambda geometries: geometries.length.sum() / METRES_PER_KM
    )
    for highway, km in by_type.sort_values(ascending=False).items():
        logger.info("  %-16s %7.0f km  (buffer %.1f m)",
                    highway, km, config.ground_truth.osm.road_buffers_m[highway])

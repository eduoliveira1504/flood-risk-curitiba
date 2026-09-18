"""Malha viária oficial de Curitiba — GeoCuritiba / IPPUC.

Substitui o OpenStreetMap como fonte de vias. Vantagens concretas sobre o OSM
neste projeto:

- **Cadastro oficial do município.** É a base do IPPUC, o mesmo instituto que o
  trabalho de referência sobre alagamento em Curitiba usou para altimetria.
- **EPSG:31982 nativo.** É exatamente o CRS de análise do projeto: nenhuma
  reprojeção da geometria viária, nenhum erro acumulado.
- **Sem limite de taxa.** O Overpass é infraestrutura voluntária que devolve 429
  sem slot e 406 por filtro anti-robô; depender dele na semana da defesa seria
  imprudente.
- **Hierarquia viária própria**, no campo ``hierarquia_viaria``, que alimenta o
  buffer por tipo.

Detalhes do ArcGIS REST que decidem a corretude:

1. **``f=json``, não ``f=geojson``.** A especificação GeoJSON obriga WGS84; pedir
   GeoJSON com ``outSR`` produz um arquivo que mente sobre o próprio CRS. O JSON
   da Esri não tem essa ambiguidade — as coordenadas estão no ``outSR`` pedido, e
   ponto.
2. **Paginação ordenada.** ``resultOffset`` sem ``orderByFields`` pode repetir ou
   pular feições entre páginas, porque a ordem não é garantida. A ordenação por
   ``objectid`` torna a paginação determinística.
3. **Valores de hierarquia desconhecidos não derrubam o estágio.** Eles recebem o
   buffer padrão e aparecem no log, para que o vocabulário real possa ser
   calibrado depois de ver o dado.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path

import requests

from .. import artifacts
from ..config import Config, StreetsConfig

logger = logging.getLogger(__name__)

__all__ = [
    "StreetsError",
    "acquire",
    "buffer_for",
    "page_params",
    "parse_features",
]

_USER_AGENT = (
    "flood-risk-curitiba/0.1 (TCC, FAE Centro Universitario, Curitiba BR; academic use)"
)
_HEADERS = {"User-Agent": _USER_AGENT, "Accept": "application/json"}

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class StreetsError(RuntimeError):
    """Falha ao consultar, paginar ou interpretar o serviço do GeoCuritiba."""


def page_params(streets: StreetsConfig, offset: int) -> dict[str, object]:
    """Parâmetros de uma página da consulta.

    ``orderByFields`` não é enfeite: sem ordem estável, ``resultOffset`` pode
    devolver a mesma feição duas vezes e pular outra.
    """
    if offset < 0:
        raise StreetsError("offset não pode ser negativo")
    return {
        "where": "1=1",
        "outFields": ",".join(streets.out_fields),
        "returnGeometry": "true",
        "outSR": streets.source_crs.split(":")[-1],
        "orderByFields": streets.order_by_field,
        "resultOffset": offset,
        "resultRecordCount": streets.page_size,
        "f": "json",
    }


def buffer_for(
    hierarchy: object, buffers: Mapping[str, float], default_m: float
) -> float:
    """Meia-largura da via, pelo valor de hierarquia.

    Comparação sem depender de caixa ou espaço em volta: o cadastro mistura
    ``"COLETORA 1"`` e ``"Coletora 1"`` entre camadas.
    """
    if hierarchy is None:
        return default_m
    key = str(hierarchy).strip().upper()
    for name, value in buffers.items():
        if name.strip().upper() == key:
            return float(value)
    return default_m


def parse_features(
    payload: Mapping,
    streets: StreetsConfig,
) -> list[dict]:
    """Converte a resposta Esri JSON em registros com geometria e largura.

    Polilinha da Esri vem como ``paths``: uma lista de partes, cada uma uma lista
    de vértices. Uma parte vira ``LineString``; várias viram ``MultiLineString``.
    """
    from shapely.geometry import LineString, MultiLineString

    if payload.get("error"):
        raise StreetsError(f"o serviço recusou a consulta: {payload['error']}")

    features = payload.get("features")
    if not isinstance(features, list):
        raise StreetsError("resposta sem a lista 'features'")

    records: list[dict] = []
    skipped = 0

    for feature in features:
        geometry = (feature or {}).get("geometry") or {}
        paths = geometry.get("paths")
        if not isinstance(paths, list) or not paths:
            skipped += 1
            continue

        parts = [
            LineString([(float(x), float(y)) for x, y, *_ in part])
            for part in paths
            if isinstance(part, list) and len(part) >= 2
        ]
        if not parts:
            skipped += 1
            continue

        attributes = feature.get("attributes") or {}
        hierarchy = attributes.get(streets.hierarchy_field)
        records.append(
            {
                "objectid": attributes.get("objectid"),
                "name": attributes.get("gtm_nm_logradouro"),
                "hierarchy": hierarchy,
                "buffer_m": buffer_for(
                    hierarchy, streets.hierarchy_buffers_m, streets.default_buffer_m
                ),
                "geometry": parts[0] if len(parts) == 1 else MultiLineString(parts),
            }
        )

    if skipped:
        logger.debug("%d feição(ões) sem geometria utilizável, descartadas", skipped)
    return records


def _fetch_page(
    streets: StreetsConfig, offset: int, session, sleep
) -> dict:
    url = f"{streets.service_url}/{streets.layer_id}/query"
    params = page_params(streets, offset)
    last_error: Exception | None = None

    for attempt in range(1, streets.max_retries + 1):
        try:
            response = session.get(
                url, params=params, timeout=streets.request_timeout_s
            )
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("tentativa %d falhou na rede: %s", attempt, exc)
        else:
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise StreetsError(
                        f"resposta do GeoCuritiba não é JSON: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise StreetsError(
                        f"esperado um objeto JSON, veio {type(payload).__name__}"
                    )
                return payload

            if response.status_code not in _RETRYABLE_STATUS:
                raise StreetsError(
                    f"HTTP {response.status_code} em {url}. Confira "
                    "'ground_truth.streets.service_url' e 'layer_id'."
                )
            last_error = StreetsError(f"HTTP {response.status_code}")
            logger.warning("tentativa %d recebeu HTTP %d", attempt, response.status_code)

        if attempt < streets.max_retries:
            backoff = 2.0 ** (attempt - 1)
            sleep(backoff)

    raise StreetsError(
        f"falhou após {streets.max_retries} tentativas em offset {offset}"
    ) from last_error


def fetch_all(
    streets: StreetsConfig, session: requests.Session | None = None, sleep=time.sleep
) -> list[dict]:
    """Percorre todas as páginas do serviço e devolve os registros."""
    http = session or requests.Session()
    for header, value in _HEADERS.items():
        http.headers.setdefault(header, value)

    records: list[dict] = []
    offset = 0
    page = 0

    while True:
        payload = _fetch_page(streets, offset, http, sleep)
        batch = parse_features(payload, streets)
        records.extend(batch)
        page += 1
        logger.info("página %d: %s feição(ões)", page, f"{len(batch):,}")

        returned = len(payload.get("features") or [])
        more = bool(payload.get("exceededTransferLimit")) or returned == streets.page_size
        if not more or returned == 0:
            break

        offset += returned
        if page > streets.max_pages:
            raise StreetsError(
                f"paginação passou de {streets.max_pages} páginas — "
                "provável laço infinito; confira 'order_by_field'"
            )

    if not records:
        raise StreetsError(
            "o serviço não devolveu nenhuma via — confira 'layer_id' "
            f"({streets.layer_id}) e o filtro da consulta"
        )
    return records


def acquire(config: Config) -> Path:
    """Estágio ``acquire-streets``: baixa a malha viária oficial do município."""
    import geopandas as gpd

    streets = config.ground_truth.streets
    destination = artifacts.streets(config)

    raw_path = config.path("data_raw") / "streets" / "geocuritiba_trecho_logradouro.json"
    if raw_path.exists():
        logger.info(
            "usando resposta em cache: %s (apague para rebaixar)",
            config.display_path(raw_path),
        )
        cached = json.loads(raw_path.read_text(encoding="utf-8"))
        records = parse_features({"features": cached}, streets)
    else:
        logger.info(
            "consultando %s camada %d (%s)",
            streets.service_url,
            streets.layer_id,
            streets.layer_name,
        )
        records = fetch_all(streets)

    frame = gpd.GeoDataFrame(records, crs=streets.source_crs)
    if frame.crs.to_string() != config.project.crs_metric:
        logger.info("reprojetando %s → %s", frame.crs, config.project.crs_metric)
        frame = frame.to_crs(config.project.crs_metric)

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(destination, driver="GPKG", layer="streets")

    _report(frame, streets, config)
    logger.info("malha viária escrita em %s", config.display_path(destination))
    return destination


def _report(frame, streets: StreetsConfig, config: Config) -> None:
    """Log da extensão por hierarquia — e do que ainda não tem buffer calibrado."""
    total_km = frame.length.sum() / 1000.0
    logger.info("trechos: %s, extensão total: %.0f km", f"{len(frame):,}", total_km)

    mapped = {k.strip().upper() for k in streets.hierarchy_buffers_m}
    grouped = frame.groupby(frame["hierarchy"].fillna("(sem valor)"), dropna=False)

    logger.info("hierarquia viária encontrada no dado:")
    unmapped: list[str] = []
    for hierarchy, part in sorted(
        grouped, key=lambda item: -item[1].length.sum()
    ):
        km = part.length.sum() / 1000.0
        buffer_m = float(part["buffer_m"].iloc[0])
        known = str(hierarchy).strip().upper() in mapped
        if not known:
            unmapped.append(str(hierarchy))
        marker = "" if known else "  <- sem buffer calibrado, usando o padrão"
        logger.info("  %-28s %7.0f km   buffer %.1f m%s", hierarchy, km, buffer_m, marker)

    if unmapped:
        logger.warning(
            "%d valor(es) de hierarquia sem buffer próprio em "
            "'ground_truth.streets.hierarchy_buffers_m': %s",
            len(unmapped),
            ", ".join(sorted(unmapped)),
        )

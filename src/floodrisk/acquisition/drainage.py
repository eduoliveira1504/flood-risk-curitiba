"""Rede de drenagem natural de Curitiba — GeoCuritiba / IPPUC, camada 47.

"Trecho de Drenagem" é o cadastro oficial dos cursos d'água do município: rios,
ribeirões e córregos, inclusive os trechos canalizados e cobertos. É mais
completa que a hidrografia em linha do IPPUC Geodownloads — toda a extensão
daquela está contida nesta, que tem cerca de 80% a mais de rede.

**O que o dado NÃO é.** Não é a rede de microdrenagem. Galeria pluvial, boca de
lobo e bueiro não estão aqui: a legenda da camada prevê um tipo "pluvial", mas
ele vem vazio. O índice enxerga para onde a água converge pelo relevo, não a
capacidade do tubo que a recebe — isso segue declarado como limitação.

**Verificação de completude.** O serviço entrega no máximo 2.000 registros por
página, e uma paginação interrompida produz um arquivo que parece inteiro. Por
isso o estágio pergunta ao serviço quantos registros existem
(``returnCountOnly``) e recusa gravar se recebeu menos.

O cliente HTTP e a paginação ordenada são os do ``acquire-streets``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path

import requests

from .. import artifacts
from ..config import Config, DrainageConfig
from . import streets as _arcgis

logger = logging.getLogger(__name__)

__all__ = [
    "DrainageError",
    "acquire",
    "check_complete",
    "parse_features",
    "summarise",
]


class DrainageError(RuntimeError):
    """Falha ao baixar, interpretar ou validar a rede de drenagem."""


def parse_features(payload: Mapping, drainage: DrainageConfig) -> list[dict]:
    """Converte a resposta Esri JSON em trechos de curso d'água.

    Só entram os trechos do tipo configurado (curso d'água). Trecho sem
    geometria utilizável é descartado em silêncio no nível de debug — acontece
    com registros de cadastro ainda sem traçado.
    """
    from shapely.geometry import LineString, MultiLineString

    if payload.get("error"):
        raise DrainageError(f"o serviço recusou a consulta: {payload['error']}")
    features = payload.get("features")
    if not isinstance(features, list):
        raise DrainageError("resposta sem a lista 'features'")

    records: list[dict] = []
    for feature in features:
        attributes = (feature or {}).get("attributes") or {}
        if attributes.get(drainage.type_field) != drainage.watercourse_type:
            continue
        paths = ((feature or {}).get("geometry") or {}).get("paths")
        if not isinstance(paths, list):
            continue
        parts = [
            LineString([(float(x), float(y)) for x, y, *_ in part])
            for part in paths
            if isinstance(part, list) and len(part) >= 2
        ]
        if not parts:
            continue
        name = (attributes.get("nome") or "").strip() or None
        records.append(
            {
                "objectid": attributes.get(drainage.order_by_field),
                "name": name,
                "covered": attributes.get(drainage.covered_field) == 1,
                "approximate": attributes.get("geometriaaproximada") == 1,
                "geometry": parts[0] if len(parts) == 1 else MultiLineString(parts),
            }
        )
    return records


def check_complete(received: int, expected: int | None) -> None:
    """Recusa um download parcial.

    ``expected`` é o que o próprio serviço diz ter. ``None`` significa que a
    contagem não pôde ser obtida — aí a única defesa é o piso de extensão total.
    """
    if expected is None:
        return
    if received < expected:
        raise DrainageError(
            f"download incompleto: o serviço tem {expected:,} trechos e chegaram "
            f"{received:,}. Uma página se perdeu; rode de novo."
        )


def _count(drainage: DrainageConfig, session) -> int | None:
    url = f"{drainage.service_url}/{drainage.layer_id}/query"
    try:
        response = session.get(
            url,
            params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
            timeout=drainage.request_timeout_s,
        )
        count = response.json().get("count")
    except (requests.RequestException, ValueError):
        return None
    return int(count) if isinstance(count, int) else None


def _fetch_all(drainage: DrainageConfig, session=None, sleep=time.sleep):
    """Todas as páginas, em Esri JSON cru, mais a contagem declarada pelo serviço."""
    http = session or requests.Session()
    for header, value in _arcgis._HEADERS.items():
        http.headers.setdefault(header, value)

    expected = _count(drainage, http)
    features: list[dict] = []
    offset = 0
    for page in range(1, drainage.max_pages + 1):
        try:
            payload = _arcgis._fetch_page(drainage, offset, http, sleep)
        except _arcgis.StreetsError as exc:
            raise DrainageError(str(exc)) from exc
        if payload.get("error"):
            raise DrainageError(f"o serviço recusou a consulta: {payload['error']}")
        batch = payload.get("features") or []
        features.extend(batch)
        logger.info("página %d: %s feição(ões)", page, f"{len(batch):,}")
        more = bool(payload.get("exceededTransferLimit")) or len(batch) == drainage.page_size
        if not more or not batch:
            break
        offset += len(batch)
    else:
        raise DrainageError(
            f"paginação passou de {drainage.max_pages} páginas — provável laço infinito"
        )

    check_complete(len(features), expected)
    return features, expected


def summarise(frame) -> dict[str, float]:
    """Extensão da rede, em km, decomposta pelo que importa ao documento."""
    length = frame.length
    return {
        "segments": float(len(frame)),
        "total_km": float(length.sum() / 1000.0),
        "covered_km": float(length[frame["covered"]].sum() / 1000.0),
        "approximate_km": float(length[frame["approximate"]].sum() / 1000.0),
        "named_km": float(length[frame["name"].notna()].sum() / 1000.0),
    }


def acquire(config: Config) -> Path:
    """Estágio ``acquire-drainage``: baixa os cursos d'água do município."""
    import geopandas as gpd

    drainage = config.drainage
    destination = artifacts.drainage(config)
    raw_path = config.path("data_raw") / "drenagem" / "geocuritiba_trecho_drenagem.json"

    if raw_path.exists():
        logger.info(
            "usando resposta em cache: %s (apague para rebaixar)",
            config.display_path(raw_path),
        )
        cached = json.loads(raw_path.read_text(encoding="utf-8"))
        features, expected = cached["features"], cached.get("service_count")
        check_complete(len(features), expected)
    else:
        logger.info(
            "consultando %s camada %d (%s)",
            drainage.service_url,
            drainage.layer_id,
            drainage.layer_name,
        )
        features, expected = _fetch_all(drainage)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(
                {
                    "service_count": expected,
                    "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "features": features,
                }
            ),
            encoding="utf-8",
        )

    records = parse_features({"features": features}, drainage)
    if not records:
        raise DrainageError("nenhum curso d'água na resposta — confira 'drainage.layer_id'")

    frame = gpd.GeoDataFrame(records, crs=drainage.source_crs)
    if frame.crs.to_string() != config.project.crs_metric:
        frame = frame.to_crs(config.project.crs_metric)

    stats = summarise(frame)
    if stats["total_km"] < drainage.min_total_length_km:
        raise DrainageError(
            f"a rede soma {stats['total_km']:.0f} km, abaixo do piso de "
            f"{drainage.min_total_length_km:.0f} km em 'drainage.min_total_length_km'. "
            "Download parcial?"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(destination, driver="GPKG", layer="drainage")

    if expected is None:
        logger.warning("o serviço não informou a contagem; completude não verificada")
    else:
        logger.info("completude: %s de %s trechos", f"{len(features):,}", f"{expected:,}")
    logger.info(
        "cursos d'água: %s trechos, %.0f km (%.0f km cobertos/canalizados, "
        "%.0f km de traçado aproximado)",
        f"{len(frame):,}",
        stats["total_km"],
        stats["covered_km"],
        stats["approximate_km"],
    )
    logger.info("rede de drenagem escrita em %s", config.display_path(destination))
    return destination

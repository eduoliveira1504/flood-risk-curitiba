"""Divisa oficial de bairros de Curitiba — GeoCuritiba / IPPUC.

Os bairros não entram no índice de suscetibilidade. Servem para resumir o
resultado na unidade que o morador e o gestor reconhecem: "quais bairros
concentram a faixa mais alta".

O cliente HTTP e a paginação ordenada são os do ``acquire-streets``. A diferença
está na geometria: bairro é polígono, e o JSON da Esri entrega polígono como
``rings`` — anéis externos em sentido horário, furos em anti-horário.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path

import requests

from .. import artifacts
from ..config import Config, NeighbourhoodsConfig
from . import streets as _arcgis

logger = logging.getLogger(__name__)

__all__ = ["NeighbourhoodsError", "acquire", "display_name", "parse_features", "polygon_from_rings"]

#: Palavras que ficam em minúscula no meio de um nome próprio em português.
_PARTICLES = frozenset({"de", "da", "do", "das", "dos", "e"})
#: Numerais romanos que aparecem em nome de bairro ("Alto da XV").
_ROMAN = frozenset(
    {"ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xix"}
)


class NeighbourhoodsError(RuntimeError):
    """Falha ao baixar, interpretar ou validar a divisa de bairros."""


def display_name(raw: object) -> str | None:
    """Nome do bairro como se escreve, a partir do cadastro em caixa alta.

    ``"CAMPO DE SANTANA"`` vira ``"Campo de Santana"``. O ``str.title`` puro
    daria "Campo De Santana".
    """
    if raw is None:
        return None
    words = str(raw).strip().lower().split()
    if not words:
        return None
    return " ".join(
        word.upper()
        if word in _ROMAN
        else word
        if index and word in _PARTICLES
        else word.capitalize()
        for index, word in enumerate(words)
    )


def _signed_area(ring) -> float:
    """Área com sinal (fórmula do cadarço): negativa para anel horário."""
    total = 0.0
    for (x1, y1), (x2, y2) in pairwise(ring):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def polygon_from_rings(rings):
    """Monta o polígono a partir dos anéis da Esri.

    Anel horário é contorno externo, anti-horário é furo. Se nenhum anel for
    horário (dado fora da convenção), todos são tratados como contorno — melhor
    um bairro sem furo do que um bairro perdido.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    clean = [
        [(float(x), float(y)) for x, y, *_ in ring]
        for ring in rings
        if isinstance(ring, list) and len(ring) >= 4
    ]
    if not clean:
        return None

    outer = [ring for ring in clean if _signed_area(ring) < 0]
    holes = [ring for ring in clean if _signed_area(ring) > 0]
    if not outer:
        outer, holes = clean, []

    shape = unary_union([Polygon(ring).buffer(0) for ring in outer])
    if holes:
        shape = shape.difference(unary_union([Polygon(ring).buffer(0) for ring in holes]))
    return None if shape.is_empty else shape


def parse_features(payload: Mapping, config: NeighbourhoodsConfig) -> list[dict]:
    """Converte a resposta Esri JSON em bairros com nome e polígono."""
    if payload.get("error"):
        raise NeighbourhoodsError(f"o serviço recusou a consulta: {payload['error']}")
    features = payload.get("features")
    if not isinstance(features, list):
        raise NeighbourhoodsError("resposta sem a lista 'features'")

    records: list[dict] = []
    for feature in features:
        attributes = (feature or {}).get("attributes") or {}
        name = display_name(attributes.get(config.name_field))
        rings = ((feature or {}).get("geometry") or {}).get("rings")
        if not name or not isinstance(rings, list):
            continue
        geometry = polygon_from_rings(rings)
        if geometry is None:
            continue
        records.append(
            {
                "name": name,
                "region": display_name(attributes.get("nm_regional")),
                "geometry": geometry,
            }
        )
    return records


def _fetch_all(config: NeighbourhoodsConfig, session=None, sleep=time.sleep) -> list[dict]:
    http = session or requests.Session()
    for header, value in _arcgis._HEADERS.items():
        http.headers.setdefault(header, value)

    features: list[dict] = []
    offset = 0
    for _ in range(config.max_pages):
        try:
            payload = _arcgis._fetch_page(config, offset, http, sleep)
        except _arcgis.StreetsError as exc:
            raise NeighbourhoodsError(str(exc)) from exc
        if payload.get("error"):
            raise NeighbourhoodsError(
                f"o serviço recusou a consulta: {payload['error']}. O GeoCuritiba "
                "fica indisponível com frequência; tente de novo em alguns minutos."
            )
        batch = payload.get("features") or []
        features.extend(batch)
        more = bool(payload.get("exceededTransferLimit")) or len(batch) == config.page_size
        if not more or not batch:
            break
        offset += len(batch)
    return features


def acquire(config: Config) -> Path:
    """Estágio ``acquire-neighbourhoods``: baixa a divisa oficial de bairros."""
    import geopandas as gpd

    settings = config.neighbourhoods
    destination = artifacts.neighbourhoods(config)

    logger.info(
        "consultando %s camada %d (%s)",
        settings.service_url,
        settings.layer_id,
        settings.layer_name,
    )
    records = parse_features({"features": _fetch_all(settings)}, settings)

    frame = gpd.GeoDataFrame(records, crs=settings.source_crs)
    if frame.crs.to_string() != config.project.crs_metric:
        frame = frame.to_crs(config.project.crs_metric)
    # Um bairro pode vir em mais de uma feição; a análise quer um polígono por nome.
    frame = frame.dissolve(by="name", aggfunc="first").reset_index()

    if len(frame) < settings.expected_count:
        raise NeighbourhoodsError(
            f"vieram {len(frame)} bairros, esperados {settings.expected_count} "
            "('neighbourhoods.expected_count'). Download parcial?"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(destination, driver="GPKG", layer="bairros")
    logger.info(
        "%d bairros, %.1f km² no total, escritos em %s",
        len(frame),
        frame.area.sum() / 1e6,
        config.display_path(destination),
    )
    return destination

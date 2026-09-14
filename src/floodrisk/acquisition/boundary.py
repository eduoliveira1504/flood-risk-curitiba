"""Limite municipal oficial de Curitiba, pela API de malhas do IBGE.

Por que isso é o primeiro estágio da Fase 1: a bounding box de Curitiba mede
~737 km² contra os ~435 km² do município. Usar a bbox como área de estudo
inflaria a estatística de impermeabilidade com Araucária, Pinhais, São José dos
Pinhais, várzea do Iguaçu e mata — e a banca perguntaria de onde saiu o número.

A bbox continua sendo a extensão de AQUISIÇÃO (ver ``geo.acquisition_geometry``);
este polígono é o que recorta os RESULTADOS.

Defesa contra mudança de API: a resposta é validada por área antes de virar
artefato. Se o IBGE mudar o contrato, ou o código do município estiver errado, o
estágio falha dizendo o que veio — em vez de gravar um polígono errado que
contaminaria todas as fases seguintes em silêncio.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from ..config import BoundaryConfig, Config

logger = logging.getLogger(__name__)

__all__ = ["BoundaryError", "acquire", "fetch_boundary", "validate_area"]

_USER_AGENT = "flood-risk-curitiba/0.1 (TCC FAE Centro Universitário; academic use)"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

SQUARE_METRES_PER_KM2 = 1_000_000.0


class BoundaryError(RuntimeError):
    """Falha ao obter ou validar o limite municipal."""


def fetch_boundary(
    boundary: BoundaryConfig,
    session: requests.Session | None = None,
    sleep=time.sleep,
) -> dict:
    """Baixa o GeoJSON do limite municipal, com repetição em falha transitória."""
    http = session or requests.Session()
    http.headers.setdefault("User-Agent", _USER_AGENT)
    last_error: Exception | None = None

    for attempt in range(1, boundary.max_retries + 1):
        try:
            response = http.get(boundary.url, timeout=boundary.request_timeout_s)
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("tentativa %d falhou na rede: %s", attempt, exc)
        else:
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise BoundaryError(
                        f"resposta do IBGE não é JSON válido: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise BoundaryError(
                        f"esperado um objeto GeoJSON, veio {type(payload).__name__}"
                    )
                return payload

            if response.status_code not in _RETRYABLE_STATUS:
                raise BoundaryError(
                    f"HTTP {response.status_code} em {boundary.url}. "
                    "Confira 'boundary.url' e 'boundary.municipality_code' "
                    "(Curitiba = 4106902)."
                )
            last_error = BoundaryError(f"HTTP {response.status_code}")
            logger.warning("tentativa %d recebeu HTTP %d", attempt, response.status_code)

        if attempt < boundary.max_retries:
            backoff = 2.0 ** (attempt - 1)
            logger.info("aguardando %.0fs", backoff)
            sleep(backoff)

    raise BoundaryError(
        f"falhou após {boundary.max_retries} tentativas em {boundary.url}"
    ) from last_error


def validate_area(area_km2: float, boundary: BoundaryConfig) -> None:
    """Recusa um polígono cuja área não corresponda ao município esperado.

    É a rede de segurança do estágio: código de município errado, mudança de
    contrato da API ou resposta parcial mudam a área em ordens de grandeza, e
    todos passariam despercebidos sem esta checagem.
    """
    low, high = boundary.area_range_km2
    if not low <= area_km2 <= high:
        raise BoundaryError(
            f"área de {area_km2:.1f} km² fora da faixa esperada "
            f"[{low:.1f}, {high:.1f}] km² para o município "
            f"{boundary.municipality_code}. O polígono baixado provavelmente não é "
            "o que se pensa. Confira 'boundary.municipality_code' e "
            "'boundary.expected_area_km2'."
        )


def acquire(config: Config) -> Path:
    """Estágio ``acquire-boundary``: baixa, valida e materializa o limite."""
    import geopandas as gpd

    boundary = config.boundary
    if boundary.source != "ibge":
        raise BoundaryError(
            f"'boundary.source' = {boundary.source!r}: apenas 'ibge' é implementado"
        )
    if not config.aoi.boundary_file:
        raise BoundaryError("'aoi.boundary_file' não está definido na configuração")

    destination = (config.root / config.aoi.boundary_file).resolve()
    if destination.exists():
        logger.info(
            "limite já existe em %s — apagando o arquivo você força o redownload",
            config.display_path(destination),
        )
        return destination

    logger.info("baixando limite municipal de %s", boundary.url)
    payload = fetch_boundary(boundary)

    # Checado antes de entregar ao geopandas: `from_features` com coleção vazia
    # levanta um ValueError sobre coluna de geometria, que não diz nada sobre a
    # causa real.
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise BoundaryError(
            "o GeoJSON do IBGE não trouxe nenhuma feição — confira "
            f"'boundary.municipality_code' ({boundary.municipality_code})"
        )

    frame = gpd.GeoDataFrame.from_features(payload, crs=boundary.source_crs)
    if frame.empty or frame.geometry.isna().all():
        raise BoundaryError("o GeoJSON do IBGE não trouxe nenhuma geometria válida")

    geometry_types = set(frame.geom_type)
    if not geometry_types <= {"Polygon", "MultiPolygon"}:
        raise BoundaryError(
            f"esperado polígono, veio {sorted(geometry_types)}"
        )

    # Um município pode vir em várias feições; o que interessa é a união.
    dissolved = frame.geometry.union_all()
    metric = gpd.GeoSeries([dissolved], crs=boundary.source_crs).to_crs(
        config.project.crs_metric
    )
    area_km2 = float(metric.area.iloc[0]) / SQUARE_METRES_PER_KM2

    logger.info("feições: %d, área: %.1f km²", len(frame), area_km2)
    validate_area(area_km2, boundary)

    # Gravado em WGS84: é o que a especificação GeoJSON pede, e `geo.aoi_geometry`
    # reprojeta na leitura.
    output = gpd.GeoDataFrame(
        {"municipality_code": [boundary.municipality_code], "area_km2": [round(area_km2, 3)]},
        geometry=[dissolved],
        crs=boundary.source_crs,
    ).to_crs(config.project.crs_geo)

    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_file(destination, driver="GeoJSON")
    logger.info("limite escrito em %s", config.display_path(destination))

    bbox_km2 = _bbox_area_km2(config)
    logger.info(
        "área de estudo caiu de %.1f km² (bbox) para %.1f km² — %.0f%% menos",
        bbox_km2,
        area_km2,
        (1 - area_km2 / bbox_km2) * 100,
    )
    return destination


def _bbox_area_km2(config: Config) -> float:
    from ..geo import acquisition_geometry

    return acquisition_geometry(config, metric=True).area / SQUARE_METRES_PER_KM2

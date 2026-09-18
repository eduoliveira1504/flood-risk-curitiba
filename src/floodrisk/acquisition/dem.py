"""Modelo digital de elevação — Copernicus DEM GLO-30.

A declividade derivada daqui é o segundo eixo do índice de suscetibilidade: é
ela que separa "impermeável e plano" — onde a água para — de "impermeável e em
ladeira", onde a água passa.

Mesmo padrão do WorldCover: leitura por janela via ``/vsicurl`` e reamostragem
para a grade de referência do mosaico Sentinel-2.

**Uma diferença que importa: reamostragem bilinear, não vizinho.** Elevação é
grandeza contínua. Vizinho mais próximo produziria degraus artificiais de 30 m
no terreno, e a declividade calculada em cima deles teria serrilhado onde o
relevo é liso — ruído que entraria direto no índice de risco.

**Honestidade sobre resolução:** o GLO-30 tem 30 m. Reamostrar para 10 m alinha
a grade, não cria detalhe. A declividade resultante carrega informação de 30 m,
e o documento deve dizer isso.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from math import floor
from pathlib import Path

from .. import artifacts
from ..config import Config, TerrainConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DEMError",
    "acquire",
    "tile_name",
    "tiles_for_bounds",
    "validate_elevation",
]

NODATA = -32768.0


class DEMError(RuntimeError):
    """Falha ao localizar, ler ou validar um tile do DEM."""


def tile_name(lat: float, lon: float) -> str:
    """Nome do tile de 1° que contém o ponto.

    Formato ``S26_00_W050_00``: hemisfério, grau do canto SUDOESTE com dois
    dígitos na latitude e três na longitude, e a parte decimal sempre ``00``
    porque os tiles são de grau inteiro.

    Como no WorldCover, o piso — e não o truncamento — é o que acerta no
    hemisfério sul e a oeste de Greenwich.
    """
    lat_corner = floor(lat)
    lon_corner = floor(lon)
    lat_hemisphere = "N" if lat_corner >= 0 else "S"
    lon_hemisphere = "E" if lon_corner >= 0 else "W"
    return (
        f"{lat_hemisphere}{abs(lat_corner):02d}_00_"
        f"{lon_hemisphere}{abs(lon_corner):03d}_00"
    )


def tiles_for_bounds(bounds: Sequence[float]) -> list[str]:
    """Todos os tiles de 1° que intersectam um envelope em WGS84."""
    west, south, east, north = bounds
    if west >= east or south >= north:
        raise DEMError(f"envelope degenerado: {tuple(bounds)}")

    names: list[str] = []
    lat = floor(south)
    while lat < north:
        lon = floor(west)
        while lon < east:
            name = tile_name(lat, lon)
            if name not in names:
                names.append(name)
            lon += 1
        lat += 1
    return names


def tile_url(terrain: TerrainConfig, tile: str) -> str:
    return terrain.url_template.format(tile=tile)


def validate_elevation(array, terrain: TerrainConfig) -> None:
    """Recusa um DEM cuja faixa de altitude não corresponda à área de estudo.

    Rede de segurança contra tile errado, unidade errada ou nodata mal tratado.
    Curitiba está no primeiro planalto paranaense, perto de 900 m; um recorte que
    devolva altitudes de nível do mar é outra coisa.
    """
    import numpy as np

    valid = array[np.isfinite(array) & (array != NODATA)]
    if valid.size == 0:
        raise DEMError("o recorte do DEM não trouxe nenhum pixel válido")

    low, high = float(valid.min()), float(valid.max())
    expected_low, expected_high = terrain.expected_elevation_range_m
    if low < expected_low or high > expected_high:
        raise DEMError(
            f"altitude entre {low:.0f} e {high:.0f} m, fora da faixa esperada "
            f"[{expected_low:.0f}, {expected_high:.0f}] m. O tile baixado "
            "provavelmente não é o da área de estudo — confira "
            "'terrain.url_template' e 'terrain.expected_elevation_range_m'."
        )
    logger.info("altitude: %.0f a %.0f m (mediana %.0f m)", low, high, float(np.median(valid)))


def acquire(config: Config) -> Path:
    """Estágio ``acquire-dem``: recorta o DEM na grade de referência."""
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject

    from ..geo import acquisition_geometry

    terrain = config.terrain
    reference = artifacts.s2_mosaic(config)
    if not reference.exists():
        raise DEMError(
            f"grade de referência ausente: {config.display_path(reference)}. "
            "Rode 'acquire-sentinel' antes."
        )

    aoi_geo = acquisition_geometry(config, metric=False)
    tiles = tiles_for_bounds(aoi_geo.bounds)
    logger.info("tiles necessários: %s", ", ".join(tiles))

    with rasterio.open(reference) as ref:
        profile = ref.profile.copy()
        target_transform = ref.transform
        target_crs = ref.crs
        target_shape = (ref.height, ref.width)

    mosaic = np.full(target_shape, np.nan, dtype="float32")

    for tile in tiles:
        url = tile_url(terrain, tile)
        logger.info("lendo %s", url)
        try:
            with rasterio.open(f"/vsicurl/{url}") as src:
                patch = np.full(target_shape, np.nan, dtype="float32")
                reproject(
                    source=rasterio.band(src, 1),
                    destination=patch,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    # Bilinear: elevação é contínua. Vizinho criaria degraus de
                    # 30 m e a declividade sairia serrilhada onde o relevo é liso.
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata if src.nodata is not None else NODATA,
                    dst_nodata=np.nan,
                )
        except rasterio.errors.RasterioIOError as exc:
            raise DEMError(
                f"não consegui abrir {url}: {exc}. Confira "
                "'terrain.url_template' e a conectividade."
            ) from exc

        mosaic = np.where(np.isnan(mosaic), patch, mosaic)

    validate_elevation(mosaic, terrain)

    profile.update(
        count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3
    )
    destination = artifacts.dem(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(mosaic, 1)
        dst.set_band_description(1, "elevation_m")
        dst.update_tags(
            source=terrain.dem_source,
            native_resolution_m="30",
            resampled_to_m=str(config.raster.resolution_m),
            resampling="bilinear",
            note=(
                "Reamostrado de 30 m para alinhar a grade de referência; "
                "nao cria detalhe de 10 m."
            ),
            tiles=",".join(tiles),
        )

    logger.info("DEM escrito em %s", config.display_path(destination))
    return destination

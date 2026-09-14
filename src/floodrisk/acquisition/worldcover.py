"""ESA WorldCover 2021 — metade do rótulo da U-Net.

A classe 50 (*built-up*) é a evidência de superfície impermeável que o
OpenStreetMap não dá: o OSM tem as vias, mas não tem telhado. Juntas, as duas
fontes cobrem o que interessa.

Duas decisões que evitam problema mais adiante:

1. **Leitura por janela remota.** Cada tile do WorldCover cobre 3° × 3° e pesa
   centenas de MB. Via ``/vsicurl`` o GDAL lê só o retângulo do AOI direto do
   bucket, sem baixar o tile inteiro.
2. **Reamostragem para a grade do mosaico Sentinel-2.** O destino não é "10 m em
   EPSG:31982", é exatamente a mesma grade de ``s2_median.tif`` — mesma
   transformação, mesma largura, mesma altura. Imagem e rótulo precisam casar
   pixel a pixel; meio pixel de deslocamento vira erro sistemático de borda que
   a U-Net aprende como se fosse sinal.

Reamostragem por vizinho mais próximo, sempre: o dado é categórico, e média de
código de classe não significa nada.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from math import floor
from pathlib import Path

from .. import artifacts
from ..config import Config, WorldCoverConfig

logger = logging.getLogger(__name__)

__all__ = [
    "WorldCoverError",
    "acquire",
    "tile_name",
    "tiles_for_bounds",
]

# Legenda oficial do WorldCover v200. Serve para validar o que foi baixado:
# qualquer valor fora daqui significa que o arquivo não é o que se pensa.
WORLDCOVER_CLASSES = {
    10: "Árvores",
    20: "Arbustos",
    30: "Vegetação herbácea",
    40: "Agricultura",
    50: "Construído",
    60: "Solo exposto / vegetação esparsa",
    70: "Neve e gelo",
    80: "Corpos d'água permanentes",
    90: "Zona úmida herbácea",
    95: "Mangue",
    100: "Musgo e liquens",
}

NODATA = 0


class WorldCoverError(RuntimeError):
    """Falha ao localizar, ler ou validar um tile do WorldCover."""


def tile_name(lat: float, lon: float, tile_size_deg: int = 3) -> str:
    """Nome do tile que contém o ponto, no padrão ``S27W051``.

    Os tiles são nomeados pelo canto SUDOESTE, alinhado a múltiplos de
    ``tile_size_deg``. Arredondar para zero em vez de para baixo é o erro
    clássico aqui, e ele só aparece no hemisfério sul ou a oeste de Greenwich —
    ou seja, exatamente em Curitiba.
    """
    if tile_size_deg < 1:
        raise WorldCoverError("tile_size_deg precisa ser positivo")

    lat_corner = floor(lat / tile_size_deg) * tile_size_deg
    lon_corner = floor(lon / tile_size_deg) * tile_size_deg

    lat_hemisphere = "N" if lat_corner >= 0 else "S"
    lon_hemisphere = "E" if lon_corner >= 0 else "W"
    return f"{lat_hemisphere}{abs(lat_corner):02d}{lon_hemisphere}{abs(lon_corner):03d}"


def tiles_for_bounds(bounds: Sequence[float], tile_size_deg: int = 3) -> list[str]:
    """Todos os tiles que intersectam um envelope em WGS84."""
    west, south, east, north = bounds
    if west >= east or south >= north:
        raise WorldCoverError(f"envelope degenerado: {tuple(bounds)}")

    lat_start = floor(south / tile_size_deg) * tile_size_deg
    lon_start = floor(west / tile_size_deg) * tile_size_deg

    names: list[str] = []
    lat = lat_start
    while lat < north:
        lon = lon_start
        while lon < east:
            name = tile_name(lat, lon, tile_size_deg)
            if name not in names:
                names.append(name)
            lon += tile_size_deg
        lat += tile_size_deg
    return names


def tile_url(worldcover: WorldCoverConfig, tile: str) -> str:
    return worldcover.url_template.format(
        version=worldcover.version, year=worldcover.year, tile=tile
    )


def acquire(config: Config) -> Path:
    """Estágio ``acquire-worldcover``: recorta o WorldCover na grade de referência."""
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject

    from ..geo import acquisition_geometry

    worldcover = config.ground_truth.worldcover
    reference = artifacts.s2_mosaic(config)
    if not reference.exists():
        raise WorldCoverError(
            f"grade de referência ausente: {config.display_path(reference)}. "
            "Rode 'acquire-sentinel' antes — é o mosaico que define o alinhamento."
        )

    aoi_geo = acquisition_geometry(config, metric=False)
    tiles = tiles_for_bounds(aoi_geo.bounds, worldcover.tile_size_deg)
    logger.info("tiles necessários: %s", ", ".join(tiles))

    with rasterio.open(reference) as ref:
        target_profile = ref.profile.copy()
        target_transform = ref.transform
        target_crs = ref.crs
        target_shape = (ref.height, ref.width)

    mosaic = np.zeros(target_shape, dtype="uint8")

    for tile in tiles:
        url = tile_url(worldcover, tile)
        logger.info("lendo %s", url)
        try:
            with rasterio.open(f"/vsicurl/{url}") as src:
                if src.crs is None or src.crs.to_epsg() != 4326:
                    raise WorldCoverError(
                        f"{tile}: esperado EPSG:4326, veio {src.crs}"
                    )
                patch = np.zeros(target_shape, dtype="uint8")
                reproject(
                    source=rasterio.band(src, 1),
                    destination=patch,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    # Categórico: média de código de classe não significa nada.
                    resampling=Resampling.nearest,
                    src_nodata=NODATA,
                    dst_nodata=NODATA,
                )
        except rasterio.errors.RasterioIOError as exc:
            raise WorldCoverError(
                f"não consegui abrir {url}: {exc}. Confira "
                "'ground_truth.worldcover.url_template' e a conectividade."
            ) from exc

        # Tiles não se sobrepõem; cada um preenche a parte que o anterior deixou.
        mosaic = np.where(mosaic == NODATA, patch, mosaic)

    validate_classes(mosaic)

    target_profile.update(
        count=1,
        dtype="uint8",
        nodata=NODATA,
        compress="deflate",
        predictor=2,
    )
    destination = artifacts.worldcover(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **target_profile) as dst:
        dst.write(mosaic, 1)
        dst.set_band_description(1, "worldcover_class")
        dst.update_tags(
            source="ESA WorldCover",
            version=worldcover.version,
            year=str(worldcover.year),
            builtup_class=str(worldcover.builtup_class),
            licence="CC-BY 4.0",
            tiles=",".join(tiles),
        )

    _report(mosaic, config)
    logger.info("WorldCover escrito em %s", config.display_path(destination))
    return destination


def validate_classes(array) -> None:
    """Recusa um raster com códigos fora da legenda do WorldCover."""
    import numpy as np

    present = set(np.unique(array).tolist()) - {NODATA}
    unknown = present - set(WORLDCOVER_CLASSES)
    if unknown:
        raise WorldCoverError(
            f"códigos fora da legenda do WorldCover: {sorted(unknown)}. "
            "O arquivo baixado provavelmente não é o produto esperado."
        )
    if not present:
        raise WorldCoverError("o recorte não trouxe nenhuma classe — AOI fora do tile?")


def _report(mosaic, config: Config) -> None:
    """Log da composição de classes, ponderada pela área válida."""

    valid = mosaic != NODATA
    total = int(valid.sum())
    if not total:
        return

    builtup = config.ground_truth.worldcover.builtup_class
    logger.info("composição no recorte (%s px válidos):", f"{total:,}")
    for code, label in WORLDCOVER_CLASSES.items():
        count = int((mosaic == code).sum())
        if count:
            marker = " <-- impermeável" if code == builtup else ""
            logger.info("  %3d %-34s %5.1f%%%s", code, label, count * 100 / total, marker)

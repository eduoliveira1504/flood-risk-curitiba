"""Caminhos canônicos dos artefatos do pipeline.

Um lugar só. Nome de arquivo repetido como string literal em vários módulos é
como se perde meia hora descobrindo que um estágio grava ``s2_median.tif`` e o
seguinte procura ``s2_mosaic.tif``.

Ler esta lista de cima a baixo também descreve o fluxo de dados do projeto.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config

__all__ = [
    "boundary",
    "impervious_mask",
    "osm_roads",
    "s2_mosaic",
    "worldcover",
]


def boundary(config: Config) -> Path:
    """Limite municipal oficial, em WGS84. Recorta os resultados."""
    if not config.aoi.boundary_file:
        raise ValueError("'aoi.boundary_file' não está definido")
    return (config.root / config.aoi.boundary_file).resolve()


def s2_mosaic(config: Config) -> Path:
    """Mosaico Sentinel-2 de mediana. É a GRADE DE REFERÊNCIA do projeto.

    Todo raster derivado — WorldCover, máscara, declividade — é alinhado a este
    arquivo. Sem uma grade de referência única, dois rasters do mesmo AOI podem
    ficar meio pixel deslocados e o cruzamento sai enviesado sem avisar.
    """
    return config.path("data_interim") / "s2_median.tif"


def worldcover(config: Config) -> Path:
    """ESA WorldCover reamostrado para a grade de referência."""
    return config.path("data_interim") / "worldcover.tif"


def osm_roads(config: Config) -> Path:
    """Malha viária do OpenStreetMap, em CRS métrico."""
    return config.path("data_interim") / "osm_roads.gpkg"


def impervious_mask(config: Config) -> Path:
    """Máscara binária de impermeabilidade — o rótulo da U-Net."""
    return config.path("data_processed") / "impervious_mask.tif"

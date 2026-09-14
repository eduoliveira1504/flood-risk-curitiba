"""Utilitários geoespaciais compartilhados por todo o pipeline.

Regra da casa: qualquer operação métrica (buffer, área, declividade, distância,
grade) acontece em ``project.crs_metric``. WGS84 só entra e sai — nunca é usado
para medir nada.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .config import Config

__all__ = [
    "Window",
    "acquisition_geometry",
    "aoi_geometry",
    "bbox_polygon",
    "grid_cells",
    "patch_windows",
    "reproject_geometry",
    "snap_bounds",
]


@dataclass(frozen=True)
class Window:
    """Janela de leitura em coordenadas de pixel (compatível com rasterio)."""

    col_off: int
    row_off: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.col_off, self.row_off, self.width, self.height)


def bbox_polygon(bounds: Sequence[float]):
    """Constrói um polígono a partir de ``[oeste, sul, leste, norte]``."""
    west, south, east, north = bounds
    return box(west, south, east, north)


def snap_bounds(
    bounds: Sequence[float], resolution: float, expand: bool = True
) -> tuple[float, float, float, float]:
    """Alinha um envelope à grade da resolução informada.

    Sem isso, dois rasters recortados do mesmo AOI podem ficar com meio pixel de
    deslocamento entre si e o cruzamento máscara × declividade sai enviesado.
    """
    if resolution <= 0:
        raise ValueError("resolution precisa ser positiva")
    west, south, east, north = bounds
    if expand:
        return (
            floor(west / resolution) * resolution,
            floor(south / resolution) * resolution,
            ceil(east / resolution) * resolution,
            ceil(north / resolution) * resolution,
        )
    return (
        ceil(west / resolution) * resolution,
        ceil(south / resolution) * resolution,
        floor(east / resolution) * resolution,
        floor(north / resolution) * resolution,
    )


def reproject_geometry(geometry: BaseGeometry, src_crs: str, dst_crs: str) -> BaseGeometry:
    """Reprojeta uma geometria shapely entre dois CRS."""
    if src_crs == dst_crs:
        return geometry

    from pyproj import Transformer
    from shapely.ops import transform

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geometry)


def acquisition_geometry(config: Config, metric: bool = True) -> BaseGeometry:
    """Extensão usada para BAIXAR dado bruto — sempre a bounding box.

    Deliberadamente ignora o limite municipal, por dois motivos:

    1. **Determinismo de cache.** O cliente do Sentinel Hub indexa a resposta por
       hash do payload. Se a extensão do pedido mudasse quando o arquivo de
       limite aparecesse no disco, o mosaico já baixado seria invalidado e a
       cota seria gasta de novo sem ninguém pedir.
    2. **Contexto de borda.** Um patch da U-Net na divisa do município precisa
       enxergar o entorno. Baixar o excedente e recortar a SAÍDA é melhor que
       recortar a entrada e treinar com borda artificial.
    """
    geometry = bbox_polygon(config.aoi.bbox)
    target = config.project.crs_metric if metric else config.project.crs_geo
    return reproject_geometry(geometry, config.project.crs_geo, target)


def aoi_geometry(config: Config, metric: bool = True) -> BaseGeometry:
    """Devolve a área de estudo.

    Usa o limite municipal oficial quando o arquivo existir; caso contrário cai
    para a bounding box da configuração. A diferença importa: a bbox de Curitiba
    inclui pedaços de municípios vizinhos.
    """
    source_crs = config.project.crs_geo
    geometry: BaseGeometry | None = None

    if config.aoi.boundary_file:
        boundary = (config.root / config.aoi.boundary_file).resolve()
        if boundary.exists():
            geometry, source_crs = _read_boundary(boundary, fallback_crs=source_crs)

    if geometry is None:
        geometry = bbox_polygon(config.aoi.bbox)

    if not metric:
        return reproject_geometry(geometry, source_crs, config.project.crs_geo)
    return reproject_geometry(geometry, source_crs, config.project.crs_metric)


def _read_boundary(path: Path, fallback_crs: str) -> tuple[BaseGeometry, str]:
    import geopandas as gpd

    frame = gpd.read_file(path)
    if frame.empty:
        raise ValueError(f"Limite municipal vazio: {path}")
    crs = frame.crs.to_string() if frame.crs is not None else fallback_crs
    return frame.geometry.union_all(), crs


def patch_windows(
    width: int, height: int, patch_size: int, overlap: int
) -> Iterator[Window]:
    """Gera janelas de patch cobrindo o raster inteiro.

    A última janela de cada linha/coluna é encostada na borda em vez de
    extrapolar, o que evita padding artificial nas bordas do município.
    """
    if patch_size <= 0:
        raise ValueError("patch_size precisa ser positivo")
    if not 0 <= overlap < patch_size:
        raise ValueError("overlap precisa estar em [0, patch_size)")
    if width < patch_size or height < patch_size:
        raise ValueError(
            f"Raster {width}x{height} é menor que o patch de {patch_size}px"
        )

    stride = patch_size - overlap
    rows = sorted({*range(0, height - patch_size + 1, stride), height - patch_size})
    cols = sorted({*range(0, width - patch_size + 1, stride), width - patch_size})

    for row in rows:
        for col in cols:
            yield Window(col_off=col, row_off=row, width=patch_size, height=patch_size)


def grid_cells(geometry: BaseGeometry, cell_size: float, clip: bool = False):
    """Grade regular métrica cobrindo a geometria.

    Devolve apenas as células que efetivamente intersectam a área de estudo.
    Com ``clip=True``, as células de borda são recortadas pelo limite.
    """
    if cell_size <= 0:
        raise ValueError("cell_size precisa ser positivo")

    west, south, east, north = snap_bounds(geometry.bounds, cell_size)
    n_cols = round((east - west) / cell_size)
    n_rows = round((north - south) / cell_size)

    cells = []
    for row in range(n_rows):
        y0 = south + row * cell_size
        for col in range(n_cols):
            x0 = west + col * cell_size
            cell = box(x0, y0, x0 + cell_size, y0 + cell_size)
            overlap = cell.intersection(geometry)
            # `intersects` é verdadeiro para células que só encostam na borda;
            # exigir área positiva descarta essas células vazias.
            if overlap.is_empty or overlap.area <= 0.0:
                continue
            cells.append(overlap if clip else cell)
    return cells

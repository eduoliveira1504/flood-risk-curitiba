"""Declividade a partir do DEM — o segundo eixo do índice de suscetibilidade.

O cruzamento do documento é ``impermeabilidade × declividade``. A impermeabilidade
diz quanta água vira escoamento; a declividade diz se essa água fica ou desce.
Superfície impermeável em terreno plano é onde o alagamento acontece; a mesma
superfície numa ladeira apenas transfere o problema para baixo.

Método de Horn (janela 3 × 3), que é o mesmo do GDAL e do ArcGIS. Não é escolha
estética: Horn pondera os vizinhos ortogonais em dobro dos diagonais, o que
suaviza ruído de DEM sem borrar quebra de relevo real. Diferença finita simples
seria mais sensível ao ruído de 30 m que estamos reamostrando.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import artifacts
from ..config import Config

logger = logging.getLogger(__name__)

__all__ = ["TerrainError", "build", "slope_degrees"]


class TerrainError(RuntimeError):
    """Insumo ausente ou inconsistente na derivação de terreno."""


def slope_degrees(elevation, cellsize_x: float, cellsize_y: float):
    """Declividade em graus, pelo método de Horn.

    A janela 3 × 3 em torno de cada pixel::

        a b c
        d e f
        g h i

    ``dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 · cellsize_x)``
    ``dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 · cellsize_y)``

    O sinal de ``dz/dy`` depende de o eixo do array crescer para o sul, mas a
    declividade usa a magnitude do gradiente — então a orientação não altera o
    resultado. (Faria diferença para a *exposição*, que não é usada aqui.)
    """
    import numpy as np

    if cellsize_x <= 0 or cellsize_y <= 0:
        raise TerrainError("o tamanho da célula precisa ser positivo")
    if elevation.ndim != 2:
        raise TerrainError(f"esperado array 2D, veio {elevation.ndim}D")
    if min(elevation.shape) < 3:
        raise TerrainError(
            f"array {elevation.shape} é pequeno demais para uma janela 3x3"
        )

    z = np.asarray(elevation, dtype="float64")
    # Borda replicada: sem isso a declividade da primeira e última linha/coluna
    # sairia como nodata e o recorte final perderia uma moldura de 10 m.
    padded = np.pad(z, 1, mode="edge")

    a = padded[:-2, :-2]
    b = padded[:-2, 1:-1]
    c = padded[:-2, 2:]
    d = padded[1:-1, :-2]
    f = padded[1:-1, 2:]
    g = padded[2:, :-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize_x)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize_y)

    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype("float32")


def build(config: Config) -> Path:
    """Estágio ``build-terrain``: deriva a declividade do DEM."""
    import numpy as np
    import rasterio

    dem_path = artifacts.dem(config)
    if not dem_path.exists():
        raise TerrainError(
            f"DEM ausente: {config.display_path(dem_path)}. Rode 'acquire-dem' antes."
        )

    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype("float32")
        profile = src.profile.copy()
        cellsize_x = abs(src.transform.a)
        cellsize_y = abs(src.transform.e)

    # Buraco no DEM viraria NaN propagado por toda a janela 3x3 em volta.
    # Preencher com a mediana mantém a declividade localmente plana ali, o que é
    # mais honesto do que espalhar nodata.
    holes = ~np.isfinite(elevation)
    if holes.any():
        logger.warning(
            "%s pixel(s) sem elevação, preenchidos com a mediana antes da derivada",
            f"{int(holes.sum()):,}",
        )
        elevation = np.where(holes, np.nanmedian(elevation), elevation)

    slope = slope_degrees(elevation, cellsize_x, cellsize_y)

    profile.update(
        count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3
    )
    destination = artifacts.slope(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(slope, 1)
        dst.set_band_description(1, "slope_degrees")
        dst.update_tags(
            method="Horn 3x3",
            units="degrees",
            source_dem=config.display_path(dem_path),
            note=(
                "DEM nativo de 30 m reamostrado para 10 m; a declividade carrega "
                "informacao de 30 m."
            ),
        )

    _report(slope, config)
    logger.info("declividade escrita em %s", config.display_path(destination))
    return destination


def _report(slope, config: Config) -> None:
    """Distribuição da declividade — o insumo do corte por quantil do índice."""
    import numpy as np

    valid = slope[np.isfinite(slope)]
    if valid.size == 0:
        return

    logger.info("declividade (graus):")
    for label, value in (
        ("mínima", float(valid.min())),
        ("p25", float(np.percentile(valid, 25))),
        ("mediana", float(np.median(valid))),
        ("p75", float(np.percentile(valid, 75))),
        ("p95", float(np.percentile(valid, 95))),
        ("máxima", float(valid.max())),
    ):
        logger.info("  %-8s %6.2f", label, value)

    flat = float((valid < 5.0).mean() * 100)
    logger.info("terreno com menos de 5 graus: %.1f%%", flat)

"""Máscara binária de superfície impermeável — o rótulo da U-Net.

    impermeável = WorldCover(classe 50)  ∪  buffer(vias do OSM)

Por que a união e não uma fonte só: o WorldCover acerta quadra construída e erra
rua — a 10 m, via estreita ladeada de árvore vira pixel misto e cai em
"árvores". O OSM tem a geometria exata dessas vias, mas não tem telhado. Cada
fonte cobre o ponto cego da outra.

No projeto anterior isso foi medido: só OSM dava ~8% de cobertura e o Dice
empacava em 0,34; com o WorldCover somado, a cobertura subiu para ~47% e o Dice
para 0,89.

Três cuidados que decidem a qualidade do rótulo:

1. **Buffer por hierarquia viária.** Uma linha de centro não tem largura. Cada
   tipo recebe a meia-largura de ``ground_truth.osm.road_buffers_m``, porque
   tratar rodovia e viela com o mesmo buffer engorda uma e some com a outra.
2. **Rasterização com ``all_touched=False``.** Contar todo pixel tocado pelo
   polígono infla sistematicamente a classe positiva na borda de cada via — num
   traçado urbano denso isso vira vários pontos percentuais de rótulo falso.
3. **Grade de referência.** A máscara sai exatamente com a transformação,
   largura e altura de ``s2_median.tif``. Imagem e rótulo precisam casar pixel a
   pixel.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import artifacts
from ..config import Config

logger = logging.getLogger(__name__)

__all__ = ["MaskError", "build", "combine", "summarise"]

IMPERVIOUS = 1
PERVIOUS = 0


class MaskError(RuntimeError):
    """Insumo ausente ou inconsistente na montagem da máscara."""


def combine(worldcover, builtup_class: int, roads_raster):
    """União lógica das duas evidências, em ``uint8``.

    Separada da leitura de arquivo de propósito: é a única regra de decisão do
    estágio, e assim pode ser testada sem tocar em disco.
    """
    import numpy as np

    if worldcover.shape != roads_raster.shape:
        raise MaskError(
            f"formas incompatíveis: WorldCover {worldcover.shape} vs "
            f"vias {roads_raster.shape}"
        )
    built = worldcover == builtup_class
    roads = roads_raster > 0
    return np.where(built | roads, IMPERVIOUS, PERVIOUS).astype("uint8")


def summarise(mask, worldcover, builtup_class: int, roads_raster) -> dict[str, float]:
    """Decompõe a máscara pela origem de cada pixel positivo.

    Serve para responder "o que o OSM acrescentou?" com número, e não com
    intuição — é exatamente a pergunta que justifica usar as duas fontes.
    """
    import numpy as np

    total = int(mask.size)
    built = worldcover == builtup_class
    roads = roads_raster > 0

    return {
        "impervious_pct": float(mask.sum()) * 100 / total,
        "worldcover_only_pct": float(np.sum(built & ~roads)) * 100 / total,
        "osm_only_pct": float(np.sum(roads & ~built)) * 100 / total,
        "both_pct": float(np.sum(built & roads)) * 100 / total,
    }


def build(config: Config) -> Path:
    """Estágio ``build-mask``: rasteriza as vias e une ao WorldCover."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    worldcover_path = artifacts.worldcover(config)
    roads_path, roads_layer = artifacts.roads(config)
    roads_stage = (
        "acquire-osm" if config.ground_truth.roads_source == "osm" else "acquire-streets"
    )

    missing = [p for p in (worldcover_path, roads_path) if not p.exists()]
    if missing:
        names = ", ".join(config.display_path(p) for p in missing)
        raise MaskError(
            f"insumo ausente: {names}. "
            f"Rode 'acquire-worldcover' e '{roads_stage}' antes."
        )

    with rasterio.open(worldcover_path) as src:
        worldcover = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        shape = (src.height, src.width)
        crs = src.crs

    roads = gpd.read_file(roads_path, layer=roads_layer)
    if roads.empty:
        raise MaskError(f"{config.display_path(roads_path)} não tem nenhuma via")
    if roads.crs is None:
        raise MaskError("a malha viária está sem CRS declarado")
    if roads.crs.to_string() != crs.to_string():
        logger.info("reprojetando vias de %s para %s", roads.crs, crs)
        roads = roads.to_crs(crs)

    # Buffer por tipo: a largura vem gravada em cada feição pelo acquire-osm.
    logger.info("bufferizando %s via(s)", f"{len(roads):,}")
    buffered = roads.geometry.buffer(roads["buffer_m"], cap_style=2)

    roads_raster = rasterize(
        ((geometry, 1) for geometry in buffered if not geometry.is_empty),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        # False de propósito: 'all_touched=True' marcaria todo pixel encostado
        # pelo polígono e inflaria a classe positiva na borda de cada via.
        all_touched=False,
    )

    mask = combine(worldcover, config.ground_truth.worldcover.builtup_class, roads_raster)
    stats = summarise(
        mask, worldcover, config.ground_truth.worldcover.builtup_class, roads_raster
    )

    profile.update(count=1, dtype="uint8", nodata=None, compress="deflate", predictor=2)
    destination = artifacts.impervious_mask(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, "impervious")
        dst.update_tags(
            definition="WorldCover classe 50 OR buffer das vias do OSM",
            worldcover_builtup_class=str(config.ground_truth.worldcover.builtup_class),
            all_touched="False",
            **{f"pct_{k}": f"{v:.2f}" for k, v in stats.items()},
        )

    logger.info("impermeável total .... %5.1f%%", stats["impervious_pct"])
    logger.info("  só WorldCover ...... %5.1f%%", stats["worldcover_only_pct"])
    logger.info("  só OSM (vias) ...... %5.1f%%  <- o que o WorldCover perdia",
                stats["osm_only_pct"])
    logger.info("  ambos .............. %5.1f%%", stats["both_pct"])
    logger.info("máscara escrita em %s", config.display_path(destination))
    return destination

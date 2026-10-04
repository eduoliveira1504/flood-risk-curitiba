"""Distância à linha de drenagem mais próxima — o terceiro fator do índice.

A declividade diz se a água fica ou desce, mas é uma medida LOCAL: um topo de
divisor de águas plano e um fundo de vale plano têm a mesma declividade, e só o
segundo recebe a água do entorno. A distância à rede de drenagem é o que separa
os dois — a rede marca as linhas para onde o relevo converge.

**A rede tem duas fontes, unidas.**

1. Os cursos d'água do cadastro do IPPUC (``acquire-drainage``): traçado
   oficial, preciso, inclusive trechos canalizados.
2. Os talvegues calculados do DEM (``flow.contributing_area``), para onde o
   cadastro é omisso. A hidrografia oficial não traz os rios enterrados do
   centro; usar só ela daria ao centro de Curitiba a menor proximidade de
   drenagem da cidade — exatamente onde o Ivo e o Belém correm sob a rua. O
   relevo guarda o vale mesmo quando o rio foi tampado.

O estágio mede a concordância entre as duas fontes onde ambas existem, que é o
que autoriza usar a segunda onde a primeira falta.

**Por que distância em planta e não altura acima da drenagem (HAND).** HAND é a
variável fisicamente mais correta, mas exige um modelo de elevação que resolva
o relevo urbano na vertical. O disponível é o Copernicus GLO-30, de 30 m e de
superfície (inclui prédio e copa): medido aqui, 10% das células ficaram com
HAND negativo. A posição do fundo de vale em planta é bem mais robusta a esse
ruído do que a diferença de cota. Trocar por HAND é a melhoria natural quando
houver MDT de curva de nível.

**Método.** Os trechos são rasterizados na grade de referência (10 m) com
``all_touched=True`` — aqui o certo é o oposto do ``build-mask``: uma linha não
tem área, e sem isso um rio em diagonal viraria pixels desconectados. Depois,
transformada de distância euclidiana exata: cada pixel recebe a distância ao
pixel de drenagem mais próximo, em metros.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import artifacts
from ..config import Config

logger = logging.getLogger(__name__)

__all__ = ["DrainageDistanceError", "build", "distance_to_true", "talweg_mask"]


class DrainageDistanceError(RuntimeError):
    """Insumo ausente ou inconsistente na derivação da distância."""


def distance_to_true(mask, resolution_m: float):
    """Distância, em metros, de cada pixel ao pixel ``True`` mais próximo.

    Pixel que já é ``True`` recebe zero.
    """
    import numpy as np
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise DrainageDistanceError(f"esperado array 2D, veio {mask.ndim}D")
    if resolution_m <= 0:
        raise DrainageDistanceError("a resolução precisa ser positiva")
    if not mask.any():
        raise DrainageDistanceError(
            "nenhum pixel de curso d'água na grade — a rede e a grade de "
            "referência estão na mesma projeção?"
        )
    return ndimage.distance_transform_edt(~mask, sampling=resolution_m).astype("float32")


def talweg_mask(elevation, resolution_m: float, native_resolution_m: float, min_area_km2: float):
    """Máscara de talvegues na grade do DEM de entrada.

    O DEM chega reamostrado para a grade de referência; aqui ele volta à
    resolução nativa por média de bloco, a área de contribuição é calculada, e a
    máscara é devolvida na grade original.
    """
    import numpy as np

    from .flow import contributing_area

    factor = max(round(native_resolution_m / resolution_m), 1)
    rows = (elevation.shape[0] // factor) * factor
    cols = (elevation.shape[1] // factor) * factor
    if rows < 3 * factor or cols < 3 * factor:
        raise DrainageDistanceError("DEM pequeno demais para a análise de escoamento")

    coarse = (
        np.asarray(elevation[:rows, :cols], dtype="float64")
        .reshape(rows // factor, factor, cols // factor, factor)
        .mean(axis=(1, 3))
    )
    cell_km2 = (resolution_m * factor) ** 2 / 1e6
    channels = contributing_area(coarse) * cell_km2 >= min_area_km2

    mask = np.zeros(elevation.shape, dtype=bool)
    mask[:rows, :cols] = np.kron(channels, np.ones((factor, factor), dtype=bool))
    return mask


def build(config: Config) -> Path:
    """Estágio ``build-drainage``: raster de distância ao curso d'água."""
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import rasterize

    network_path = artifacts.drainage(config)
    reference_path = artifacts.s2_mosaic(config)
    if not reference_path.exists():
        # A declividade está na mesma grade e serve de referência equivalente.
        reference_path = artifacts.slope(config)
    missing = [p for p in (network_path, reference_path) if not p.exists()]
    if missing:
        names = ", ".join(config.display_path(p) for p in missing)
        raise DrainageDistanceError(
            f"insumo ausente: {names}. Rode 'acquire-drainage' antes."
        )

    with rasterio.open(reference_path) as source:
        profile = source.profile.copy()
        transform = source.transform
        shape = (source.height, source.width)
        crs = source.crs
        resolution = abs(source.transform.a)

    network = gpd.read_file(network_path, layer="drainage")
    if network.empty:
        raise DrainageDistanceError(f"{config.display_path(network_path)} está vazio")
    if network.crs is None:
        raise DrainageDistanceError("a rede de drenagem está sem CRS declarado")
    if network.crs.to_string() != crs.to_string():
        network = network.to_crs(crs)

    channels = rasterize(
        ((geometry, 1) for geometry in network.geometry if not geometry.is_empty),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    cadastre = channels == 1

    dem_path = artifacts.dem(config)
    if not dem_path.exists():
        raise DrainageDistanceError(
            f"DEM ausente: {config.display_path(dem_path)}. Rode 'acquire-dem' antes."
        )
    with rasterio.open(dem_path) as source:
        elevation = source.read(1).astype("float64")
    if elevation.shape != shape:
        raise DrainageDistanceError("o DEM está em grade diferente da referência")
    holes = ~np.isfinite(elevation)
    if holes.any():
        elevation = np.where(holes, np.nanmedian(elevation), elevation)

    talwegs = talweg_mask(
        elevation,
        resolution,
        config.drainage.dem_native_resolution_m,
        config.drainage.talweg_min_area_km2,
    )
    stats = _agreement(cadastre, talwegs, resolution, config)
    distance = distance_to_true(cadastre | talwegs, resolution)

    profile.update(
        count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3
    )
    destination = artifacts.drainage_distance(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(distance, 1)
        dst.set_band_description(1, "distance_to_watercourse_m")
        dst.update_tags(
            method="distancia euclidiana exata a rede rasterizada (all_touched)",
            units="metres",
            source=config.display_path(network_path),
            segments=str(len(network)),
            network_km=f"{network.length.sum() / 1000.0:.1f}",
            talweg_min_area_km2=str(config.drainage.talweg_min_area_km2),
            **{key: f"{value:.2f}" for key, value in stats.items()},
        )

    logger.info("distância à drenagem (m), sobre a grade inteira:")
    for label, q in (("p25", 25), ("mediana", 50), ("p75", 75), ("p95", 95)):
        logger.info("  %-8s %7.0f", label, float(np.percentile(distance, q)))
    logger.info("  %-8s %7.0f", "máxima", float(distance.max()))
    logger.info("distância escrita em %s", config.display_path(destination))
    return destination


def _agreement(cadastre, talwegs, resolution: float, config: Config) -> dict[str, float]:
    """Quanto o talvegue do DEM coincide com o rio do cadastro, dentro do município.

    É a verificação que sustenta o complemento: se o relevo acha o vale onde o
    cadastro tem rio, é razoável confiar nele onde o cadastro não tem. A
    referência de acaso vai junto — o cadastro é denso, e sem ela "80% a menos
    de 100 m" não diria nada.
    """
    import rasterio
    from rasterio.features import rasterize

    from ..geo import aoi_geometry

    with rasterio.open(artifacts.dem(config)) as source:
        inside = (
            rasterize(
                [(aoi_geometry(config, metric=True), 1)],
                out_shape=cadastre.shape,
                transform=source.transform,
                dtype="uint8",
            )
            == 1
        )

    to_cadastre = distance_to_true(cadastre, resolution)
    tolerance_m = 100.0
    talweg_inside = talwegs & inside
    stats = {
        "talweg_near_cadastre_pct": float(
            100 * (to_cadastre[talweg_inside] <= tolerance_m).mean()
        ),
        "chance_near_cadastre_pct": float(100 * (to_cadastre[inside] <= tolerance_m).mean()),
        "talweg_km": float(talweg_inside.sum() * resolution / 1000.0)
        / max(round(config.drainage.dem_native_resolution_m / resolution), 1),
        "talweg_new_km": float(
            (talweg_inside & (to_cadastre > tolerance_m)).sum() * resolution / 1000.0
        )
        / max(round(config.drainage.dem_native_resolution_m / resolution), 1),
        "max_gap_cadastre_only_m": float(to_cadastre[inside].max()),
        "max_gap_combined_m": float(
            distance_to_true(cadastre | talwegs, resolution)[inside].max()
        ),
    }
    logger.info(
        "talvegues do DEM (área de contribuição >= %.2f km²): %.0f km no município",
        config.drainage.talweg_min_area_km2,
        stats["talweg_km"],
    )
    logger.info(
        "  %.0f%% deles a até %.0f m de um curso d'água do cadastro (ao acaso "
        "seriam %.0f%%)",
        stats["talweg_near_cadastre_pct"],
        tolerance_m,
        stats["chance_near_cadastre_pct"],
    )
    logger.info(
        "  %.0f km são linha nova, a mais de %.0f m de qualquer rio cadastrado",
        stats["talweg_new_km"],
        tolerance_m,
    )
    logger.info(
        "  maior distância a uma drenagem dentro do município: %.0f m só com o "
        "cadastro, %.0f m com o complemento",
        stats["max_gap_cadastre_only_m"],
        stats["max_gap_combined_m"],
    )
    return stats

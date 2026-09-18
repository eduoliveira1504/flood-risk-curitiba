"""Gera a camada de satélite do site a partir do próprio mosaico Sentinel-2.

Por que não usar um basemap de satélite de terceiro: o mosaico que o site exibe
passa a ser **exatamente o mesmo dado que treinou a rede** — mesma data, mesma
composição, mesma resolução. Quem olhar o mapa está vendo a entrada do modelo,
não uma imagem qualquer de outro ano e outro sensor. Além de honesto, elimina
uma dependência externa que pode exigir chave, cobrar ou sair do ar.

Três conversões acontecem aqui, e cada uma tem um motivo:

**Reprojeção para Web Mercator (EPSG:3857).** É a projeção em que o Leaflet
desenha. Com a imagem já nela, um ``imageOverlay`` com os cantos corretos encaixa
pixel a pixel no mapa. Servir o GeoTIFF em SIRGAS 2000 e deixar o navegador
"esticar" produziria um deslocamento que cresce para as bordas — e a célula de
suscetibilidade apareceria fora do quarteirão a que pertence.

**Esticamento por percentil.** Reflectância de superfície ocupa uma fração
pequena da faixa de 16 bits; mostrada crua, a imagem sai quase preta. O corte em
p2–p98 por banda usa a faixa que o dado realmente ocupa. Os percentis ignoram o
nodata, senão o zero das bordas puxaria o limite inferior e lavaria a imagem.

**JPEG, não PNG.** Imagem de satélite é fotográfica: o JPEG comprime uma ordem de
grandeza melhor que o PNG no mesmo material, e artefato de compressão em imagem
de fundo é invisível na prática. PNG só valeria a pena pela transparência, que
aqui não é necessária — a bbox é retangular e o entorno do município é contexto
útil, não ruído a esconder.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import artifacts
from .config import Config

logger = logging.getLogger(__name__)

__all__ = ["BasemapError", "build_basemap", "stretch_to_byte"]

#: Maior lado da imagem publicada, em pixels. O mosaico tem 3.484 px de altura;
#: publicar nesse tamanho custa megabytes para um detalhe que o zoom do mapa
#: municipal não mostra.
MAX_SIDE_PX = 2048

#: Qualidade JPEG. Acima de ~85 o arquivo cresce sem ganho visível.
JPEG_QUALITY = 85

#: Percentis do esticamento de contraste.
STRETCH_PERCENTILES = (2.0, 98.0)


class BasemapError(RuntimeError):
    """Insumo ausente ou inconsistente na geração da camada de satélite."""


def stretch_to_byte(band, low: float, high: float):
    """Converte uma banda para 0–255 recortando na faixa informada.

    ``low >= high`` acontece quando a banda é constante — recorte de água
    uniforme, por exemplo. Nesse caso devolve cinza médio em vez de estourar numa
    divisão por zero.
    """
    import numpy as np

    array = np.asarray(band, dtype="float32")
    if not high > low:
        return np.full(array.shape, 128, dtype="uint8")
    clipped = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (clipped * 255).astype("uint8")


def build_basemap(
    config: Config,
    destination: Path,
    max_side: int = MAX_SIDE_PX,
    quality: int = JPEG_QUALITY,
) -> dict:
    """Escreve o JPEG em Web Mercator e devolve os limites em latitude/longitude.

    Os limites voltam no formato que o Leaflet espera em ``imageOverlay``:
    ``[[sul, oeste], [norte, leste]]``.
    """
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

    source_path = artifacts.s2_mosaic(config)
    if not source_path.exists():
        raise BasemapError(
            f"mosaico ausente: {config.display_path(source_path)}. "
            "Rode 'acquire-sentinel' antes."
        )

    with rasterio.open(source_path) as source:
        if source.count < 3:
            raise BasemapError(
                f"o mosaico tem {source.count} banda(s); são necessárias ao menos 3"
            )

        target_crs = "EPSG:3857"
        transform, width, height = calculate_default_transform(
            source.crs, target_crs, source.width, source.height, *source.bounds
        )

        # Reduz já na reprojeção, em vez de reprojetar cheio e encolher depois:
        # metade do trabalho e da memória, mesmo resultado visual.
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            width = max(int(width * scale), 1)
            height = max(int(height * scale), 1)
            transform, width, height = calculate_default_transform(
                source.crs,
                target_crs,
                source.width,
                source.height,
                *source.bounds,
                dst_width=width,
                dst_height=height,
            )

        # B04, B03, B02 — a ordem de bandas vem da configuração, e as três
        # primeiras são azul, verde e vermelho.
        channels = []
        for index in (3, 2, 1):
            destination_band = np.zeros((height, width), dtype="uint16")
            reproject(
                source=rasterio.band(source, index),
                destination=destination_band,
                dst_transform=transform,
                dst_crs=target_crs,
                # Bilinear: a imagem está sendo reduzida, e vizinho mais próximo
                # produziria serrilhado em telhado e traçado viário.
                resampling=Resampling.bilinear,
                src_nodata=source.nodata if source.nodata is not None else 0,
                dst_nodata=0,
            )
            channels.append(destination_band)

        source_bounds = source.bounds
        source_crs = source.crs

    stacked = []
    for band in channels:
        valid = band[band > 0]
        if valid.size == 0:
            raise BasemapError(
                "o mosaico não tem pixel válido depois da reprojeção — "
                "a extensão está correta?"
            )
        low, high = np.percentile(valid, STRETCH_PERCENTILES)
        stacked.append(stretch_to_byte(band, float(low), float(high)))

    image = Image.fromarray(np.dstack(stacked), mode="RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=quality, optimize=True, progressive=True)

    west, south, east, north = transform_bounds(source_crs, "EPSG:4326", *source_bounds)
    logger.info(
        "camada de satélite: %d × %d px, %.0f kB",
        width,
        height,
        destination.stat().st_size / 1000,
    )
    return {
        "file": destination.name,
        "bounds": [[south, west], [north, east]],
        "width": width,
        "height": height,
        "source": "Copernicus Sentinel-2 L2A — mediana temporal",
        "period": f"{config.sentinel.date_start} a {config.sentinel.date_end}",
    }

"""Estágio ``infer``: aplica a U-Net à cidade inteira.

O treino trabalha com patches de 128 px porque é assim que a rede aprende. O
produto precisa de um raster contínuo de 2176 × 3484 px. Costurar um a partir do
outro tem uma armadilha conhecida: **a borda do patch é onde a predição é pior**.
O campo receptivo de um pixel no canto da janela é metade do de um pixel no
centro — ele simplesmente viu menos contexto. Recortar patches encostados um no
outro e colar produz uma grade de costura visível, com artefato exatamente a
cada 128 px, e num mapa de risco isso aparece como faixas de risco que seguem a
grade de processamento em vez do território.

A solução aqui tem duas partes:

1. **Sobreposição maior que a do treino.** Cada pixel é predito por várias
   janelas, em posições diferentes dentro de cada uma.
2. **Média ponderada por janela de Hann.** O peso de cada predição decai do
   centro do patch para a borda, então a opinião do pixel que viu contexto
   completo domina a do que viu metade. O resultado é contínuo por construção.

O piso nos pesos não é detalhe estético: nas bordas do raster completo existem
pixels cobertos por uma única janela, e justamente na região de peso baixo dela.
Sem piso, a soma de pesos ali tende a zero e a divisão final explode.

A saída é o raster de probabilidade (que preserva a incerteza para o índice de
suscetibilidade usar) e a binarização no limiar calibrado. Fora da área com dado
válido, ambos saem como nodata — pixel sem cena limpa não é área permeável.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import artifacts
from ..config import Config

logger = logging.getLogger(__name__)

__all__ = ["InferError", "blend_weights", "run"]

NODATA_PROBABILITY = -1.0
NODATA_CLASS = 255

#: Piso dos pesos de mistura, como fração do máximo. Ver o docstring do módulo.
WEIGHT_FLOOR = 0.05


class InferError(RuntimeError):
    """Insumo ausente ou incompatível na inferência."""


def blend_weights(size: int):
    """Janela de Hann 2D com piso, para a média ponderada entre patches.

    ``np.hanning(size)`` zera nas duas pontas. Usa-se ``size + 2`` e descartam-se
    as extremidades para que nenhum pixel do patch entre com peso exatamente
    zero — um pixel de borda do raster pode ser coberto só por essa janela.
    """
    import numpy as np

    if size < 2:
        raise InferError("a janela de mistura precisa de ao menos 2 pixels")

    profile = np.hanning(size + 2)[1:-1]
    weights = np.outer(profile, profile).astype("float32")
    return np.maximum(weights, WEIGHT_FLOOR * weights.max())


def run(config: Config) -> Path:
    """Aplica o modelo ao mosaico inteiro e grava probabilidade e classe."""
    import numpy as np
    import rasterio
    import torch

    from ..geo import patch_windows
    from .data import REFLECTANCE_SCALE, Normalisation
    from .train import build_model

    mosaic_path = artifacts.s2_mosaic(config)
    if not mosaic_path.exists():
        raise InferError(
            f"mosaico ausente: {config.display_path(mosaic_path)}. "
            "Rode 'acquire-sentinel' antes."
        )

    checkpoint_path = artifacts.checkpoint(config)
    if not checkpoint_path.exists():
        raise InferError(
            f"checkpoint ausente: {config.display_path(checkpoint_path)}. "
            "Rode 'train' antes."
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = Normalisation.from_dict(payload["normalisation"])
    mean = np.asarray(stats.mean, dtype="float32").reshape(-1, 1, 1)
    std = np.asarray(stats.std, dtype="float32").reshape(-1, 1, 1)

    patch_size = payload.get("patch_size", config.raster.patch_size)
    overlap = config.raster.inference_overlap
    stride = patch_size - overlap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    with rasterio.open(mosaic_path) as source:
        profile = source.profile.copy()
        height, width = source.height, source.width
        bands = source.read().astype("float32")

    if bands.shape[0] != config.model.in_channels:
        raise InferError(
            f"o mosaico tem {bands.shape[0]} bandas e o modelo espera "
            f"{config.model.in_channels}"
        )

    # Um pixel sem nenhuma cena limpa saiu 0 em todas as bandas — mesma convenção
    # do 'make-dataset'.
    valid = (bands > 0).any(axis=0)
    normalised = np.where(valid, (bands / REFLECTANCE_SCALE - mean) / std, 0.0)
    normalised = normalised.astype("float32")

    accumulated = np.zeros((height, width), dtype="float32")
    weight_total = np.zeros((height, width), dtype="float32")
    weights = blend_weights(patch_size)

    windows = list(patch_windows(width, height, patch_size, overlap))
    logger.info(
        "%d janelas de %dpx, passo %dpx, sobreposição %dpx",
        len(windows),
        patch_size,
        stride,
        overlap,
    )

    batch_size = config.model.batch_size
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size]
            stack = np.stack(
                [
                    normalised[
                        :,
                        window.row_off : window.row_off + window.height,
                        window.col_off : window.col_off + window.width,
                    ]
                    for window in chunk
                ]
            )
            logits = model(torch.from_numpy(stack).to(device))
            probability = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]

            for window, prediction in zip(chunk, probability, strict=True):
                rows = slice(window.row_off, window.row_off + window.height)
                cols = slice(window.col_off, window.col_off + window.width)
                accumulated[rows, cols] += prediction * weights
                weight_total[rows, cols] += weights

            if start % (batch_size * 20) == 0:
                logger.info("  %d/%d janelas", min(start + batch_size, len(windows)), len(windows))

    if (weight_total <= 0).any():
        raise InferError(
            "há pixels que nenhuma janela cobriu — 'raster.inference_overlap' "
            "ou o tamanho do patch estão incoerentes com o mosaico."
        )

    probability = accumulated / weight_total
    threshold = payload.get(
        "binarization_threshold", config.evaluation.binarization_threshold
    )
    predicted = (probability >= threshold).astype("uint8")

    probability = np.where(valid, probability, NODATA_PROBABILITY).astype("float32")
    predicted = np.where(valid, predicted, NODATA_CLASS).astype("uint8")

    probability_path = _write(
        config,
        artifacts.impervious_probability(config),
        probability,
        profile,
        dtype="float32",
        nodata=NODATA_PROBABILITY,
        description="impervious_probability",
        tags={
            "model": f"{config.model.arch}/{config.model.encoder}",
            "checkpoint_epoch": str(payload.get("epoch")),
            "blend": "hann2d",
            "inference_overlap_px": str(overlap),
        },
    )
    _write(
        config,
        artifacts.impervious_predicted(config),
        predicted,
        profile,
        dtype="uint8",
        nodata=NODATA_CLASS,
        description="impervious_predicted",
        tags={"threshold": f"{threshold:.2f}", "nodata": str(NODATA_CLASS)},
    )

    _report(config, probability, predicted, valid, threshold)
    return probability_path


def _write(config, destination, array, profile, dtype, nodata, description, tags):
    import rasterio

    profile = dict(profile)
    profile.update(
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype == "float32" else 2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as sink:
        sink.write(array, 1)
        sink.set_band_description(1, description)
        sink.update_tags(**tags)
    logger.info("%s escrito em %s", description, config.display_path(destination))
    return destination


def _report(config: Config, probability, predicted, valid, threshold: float) -> None:
    """Compara a predição com o rótulo em toda a cidade.

    Não é uma métrica de desempenho — o rótulo cobre o mesmo território que o
    treino viu, então isto não estima generalização nenhuma. Serve para outra
    coisa: mostrar **onde e quanto** o modelo discorda da fonte que o treinou.
    Concordância perto de 100% significaria que a rede virou uma cópia do
    WorldCover e não acrescenta informação ao índice.
    """
    import numpy as np
    import rasterio

    total = int(np.count_nonzero(valid))
    positives = int(np.count_nonzero((predicted == 1) & valid))
    logger.info("impermeável predito: %.1f%% da área com dado", 100 * positives / total)
    logger.info(
        "probabilidade indecisa (0,2–0,8): %.1f%% dos pixels válidos",
        100
        * np.count_nonzero((probability > 0.2) & (probability < 0.8) & valid)
        / total,
    )

    mask_path = artifacts.impervious_mask(config)
    if not mask_path.exists():
        return

    with rasterio.open(mask_path) as source:
        reference = source.read(1)
    if reference.shape != predicted.shape:
        logger.warning("máscara e predição em grades diferentes; comparação omitida")
        return

    agreement = (predicted == reference) & valid
    both = (predicted == 1) & (reference == 1) & valid
    union = ((predicted == 1) | (reference == 1)) & valid

    logger.info(
        "concordância com o rótulo: %.1f%% dos pixels · IoU %.4f",
        100 * np.count_nonzero(agreement) / total,
        np.count_nonzero(both) / max(np.count_nonzero(union), 1),
    )
    logger.info(
        "o modelo acrescenta %.1f%% e retira %.1f%% de área impermeável "
        "em relação ao rótulo",
        100 * np.count_nonzero((predicted == 1) & (reference == 0) & valid) / total,
        100 * np.count_nonzero((predicted == 0) & (reference == 1) & valid) / total,
    )
    logger.info("limiar aplicado: %.2f", threshold)

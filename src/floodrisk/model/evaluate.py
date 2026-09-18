"""Estágio ``evaluate``: gasta o conjunto de teste, uma vez.

Este é o único lugar do projeto autorizado a olhar o teste, e é por isso que ele
existe separado do ``train``. A regra que ele protege é simples de enunciar e
fácil de violar sem perceber: **nenhuma decisão pode ser tomada olhando o
teste**. Nem escolha de época, nem de hiperparâmetro, nem de limiar.

O limiar de binarização é onde quase todo mundo escorrega. Varrer limiares e
reportar o melhor resultado de teste é escolher o resultado — a métrica deixa de
ser uma estimativa de desempenho e vira o máximo de uma busca. Aqui a varredura
acontece **no conjunto de validação**, e o limiar que ela devolve é então
aplicado ao teste sem mais nenhum ajuste. O relatório traz as duas leituras (no
limiar fixo de 0,5 e no limiar calibrado na validação) justamente para que a
diferença entre elas fique visível em vez de escondida.

A figura não mostra patches sorteados. Mostra os de **pior** Dice, porque é onde
a falha aparece: um mosaico de acertos bonitos não informa nada que a métrica já
não tenha dito, e esconde exatamente o caso que a banca vai perguntar.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .. import artifacts
from ..config import Config

logger = logging.getLogger(__name__)

__all__ = ["EvaluateError", "best_threshold", "counts_at", "per_patch_dice", "run"]

#: Limiares varridos na validação. Passo de 0,01 entre 0,05 e 0,95 — fora disso
#: a predição degenera para tudo-positivo ou tudo-negativo.
THRESHOLD_GRID = [round(0.05 + 0.01 * step, 2) for step in range(91)]


class EvaluateError(RuntimeError):
    """Checkpoint ausente ou incompatível com a configuração atual."""


# --------------------------------------------------------------------------- #
# Contagens — NumPy puro, sem PyTorch, para poderem ser testadas
# --------------------------------------------------------------------------- #


def counts_at(probability, label, valid, threshold: float):
    """VP, FP, FN e VN sobre os pixels válidos, a um dado limiar."""
    import numpy as np

    keep = valid > 0
    predicted = (probability >= threshold) & keep
    reference = (label > 0) & keep

    true_positive = float(np.count_nonzero(predicted & reference))
    false_positive = float(np.count_nonzero(predicted & ~reference & keep))
    false_negative = float(np.count_nonzero(~predicted & reference & keep))
    true_negative = float(np.count_nonzero(~predicted & ~reference & keep))
    return true_positive, false_positive, false_negative, true_negative


def dice_at(probability, label, valid, threshold: float) -> float:
    tp, fp, fn, _ = counts_at(probability, label, valid, threshold)
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def best_threshold(probability, label, valid, grid=None) -> tuple[float, float]:
    """Limiar de maior Dice na grade. Chamado SÓ com o conjunto de validação.

    Empates ficam com o limiar mais baixo da grade, por escolha declarada: num
    mapa de risco, deixar de marcar superfície impermeável é o erro mais caro
    dos dois, então na dúvida o critério favorece recall.
    """
    # `grid or THRESHOLD_GRID` cairia no padrão diante de uma lista vazia, em
    # silêncio — e uma grade vazia é erro de chamada, não pedido de padrão.
    grid = list(THRESHOLD_GRID if grid is None else grid)
    if not grid:
        raise EvaluateError("grade de limiares vazia")

    scored = [(dice_at(probability, label, valid, value), -value) for value in grid]
    best_dice, negative_threshold = max(scored)
    return -negative_threshold, best_dice


def per_patch_dice(probability, label, valid, threshold: float):
    """Dice de cada patch isoladamente, para achar os piores casos.

    Patch sem nenhum pixel positivo no rótulo E sem nenhum positivo na predição
    recebe 1.0: acertar "não há nada aqui" é acerto, e devolver 0 nesse caso
    colocaria os acertos perfeitos no topo da lista de piores.
    """
    import numpy as np

    scores = np.empty(probability.shape[0], dtype="float64")
    for index in range(probability.shape[0]):
        tp, fp, fn, _ = counts_at(
            probability[index], label[index], valid[index], threshold
        )
        denominator = 2 * tp + fp + fn
        scores[index] = (2 * tp / denominator) if denominator else 1.0
    return scores


def metrics_at(probability, label, valid, threshold: float) -> dict[str, float]:
    """Painel completo a um limiar, incluindo a matriz de confusão bruta."""
    from .loss import segmentation_metrics

    tp, fp, fn, tn = counts_at(probability, label, valid, threshold)
    metrics = segmentation_metrics(tp, fp, fn)
    total = tp + fp + fn + tn
    metrics.update(
        {
            "threshold": threshold,
            "accuracy": ((tp + tn) / total) if total else 0.0,
            "specificity": (tn / (tn + fp)) if (tn + fp) else 0.0,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "valid_pixels": total,
            "positive_rate": ((tp + fn) / total) if total else 0.0,
        }
    )
    return metrics


# --------------------------------------------------------------------------- #
# Inferência sobre um conjunto
# --------------------------------------------------------------------------- #


def _load_checkpoint(config: Config):
    import torch

    path = artifacts.checkpoint(config)
    if not path.exists():
        raise EvaluateError(
            f"checkpoint ausente: {config.display_path(path)}. Rode 'train' antes."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    stored = payload.get("patch_size")
    if stored is not None and stored != config.raster.patch_size:
        raise EvaluateError(
            f"o checkpoint foi treinado com patch de {stored}px e a configuração "
            f"atual pede {config.raster.patch_size}px. Refaça 'make-dataset' e "
            "'train', ou avalie com a configuração original."
        )
    return payload


def _predict(config: Config, payload, split: str):
    """Probabilidades, rótulos e validade de um conjunto inteiro, em NumPy."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from .data import Normalisation, PatchDataset, load_manifest
    from .train import build_model

    rows = load_manifest(config, split)
    stats = Normalisation.from_dict(payload["normalisation"])
    dataset = PatchDataset(config, rows, stats, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=config.model.batch_size,
        shuffle=False,
        num_workers=config.model.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    probabilities, labels, valids = [], [], []
    with torch.no_grad():
        for image, label, valid in loader:
            logits = model(image.to(device))
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy()[:, 0])
            labels.append(label.numpy()[:, 0])
            valids.append(valid.numpy()[:, 0])

    return (
        np.concatenate(probabilities),
        np.concatenate(labels),
        np.concatenate(valids),
        dataset.ids,
        rows,
    )


def run(config: Config) -> Path:
    """Avalia na validação, calibra o limiar nela, e só então toca no teste."""
    payload = _load_checkpoint(config)
    fixed = config.evaluation.binarization_threshold

    logger.info("inferindo na validação")
    val_prob, val_label, val_valid, _, _ = _predict(config, payload, "val")

    tuned, tuned_val_dice = best_threshold(val_prob, val_label, val_valid)
    logger.info(
        "limiar calibrado NA VALIDAÇÃO: %.2f (Dice %.4f) — fixo %.2f dá Dice %.4f",
        tuned,
        tuned_val_dice,
        fixed,
        dice_at(val_prob, val_label, val_valid, fixed),
    )

    logger.info("inferindo no TESTE — este conjunto é gasto agora, uma vez")
    test_prob, test_label, test_valid, test_ids, test_rows = _predict(
        config, payload, "test"
    )

    report = {
        "checkpoint_epoch": payload.get("epoch"),
        "encoder": config.model.encoder,
        "patch_size": config.raster.patch_size,
        "fixed_threshold": fixed,
        "tuned_threshold": tuned,
        "tuned_on": "val",
        "validation": {
            "fixed": metrics_at(val_prob, val_label, val_valid, fixed),
            "tuned": metrics_at(val_prob, val_label, val_valid, tuned),
        },
        "test": {
            "fixed": metrics_at(test_prob, test_label, test_valid, fixed),
            "tuned": metrics_at(test_prob, test_label, test_valid, tuned),
        },
    }

    destination = artifacts.evaluation_report(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)

    _report(report, config)
    _plot_worst(config, test_prob, test_label, test_valid, test_ids, test_rows, tuned)

    logger.info("relatório escrito em %s", config.display_path(destination))
    return destination


def _report(report, config: Config) -> None:
    logger.info(
        "%-12s %-8s %8s %8s %10s %8s",
        "conjunto",
        "limiar",
        "dice",
        "iou",
        "precisão",
        "recall",
    )
    for split in ("validation", "test"):
        for kind in ("fixed", "tuned"):
            metrics = report[split][kind]
            logger.info(
                "%-12s %-8.2f %8.4f %8.4f %10.4f %8.4f",
                split,
                metrics["threshold"],
                metrics["dice"],
                metrics["iou"],
                metrics["precision"],
                metrics["recall"],
            )

    test_dice = report["test"]["tuned"]["dice"]
    val_dice = report["validation"]["tuned"]["dice"]
    gap = val_dice - test_dice
    logger.info("diferença validação − teste: %+.4f", gap)
    if gap > 0.05:
        logger.warning(
            "o teste ficou %.3f abaixo da validação. Diferença dessa ordem num "
            "split espacial costuma ser território, não sobreajuste: confira a "
            "prevalência de cada conjunto em patches/split.json antes de mexer "
            "no modelo.",
            gap,
        )
    if test_dice < config.evaluation.dice_threshold:
        logger.warning(
            "Dice de teste %.4f abaixo do limiar de %.2f declarado no documento.",
            test_dice,
            config.evaluation.dice_threshold,
        )


def _plot_worst(config, probability, label, valid, ids, rows, threshold, count=6):
    """Os patches de pior Dice, lado a lado: imagem, rótulo, predição e erro."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    from .data import REFLECTANCE_SCALE

    scores = per_patch_dice(probability, label, valid, threshold)
    order = np.argsort(scores)[: min(count, len(scores))]

    figure, axes = plt.subplots(len(order), 4, figsize=(11, 2.8 * len(order)))
    axes = np.atleast_2d(axes)

    # Vermelho = falso positivo (marcou impermeável onde não é);
    # azul = falso negativo (deixou de marcar). Cinza = acerto.
    error_colours = ListedColormap(["#e8e8e8", "#d62728", "#1f77b4"])

    for row_index, patch_index in enumerate(order):
        source = rows[patch_index]
        path = Path(source["file"])
        path = path if path.is_absolute() else (config.root / path)
        with np.load(path) as stored:
            bands = stored["image"].astype("float32") / REFLECTANCE_SCALE

        # B04, B03, B02 — as bandas estão na ordem da configuração.
        rgb = np.dstack([bands[2], bands[1], bands[0]])
        low, high = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)

        reference = label[patch_index] > 0
        predicted = probability[patch_index] >= threshold
        keep = valid[patch_index] > 0

        error = np.zeros(reference.shape, dtype="uint8")
        error[predicted & ~reference & keep] = 1
        error[~predicted & reference & keep] = 2

        for column, (image, cmap, title) in enumerate(
            [
                (rgb, None, f"{ids[patch_index]}\nDice {scores[patch_index]:.3f}"),
                (reference, "gray", "rótulo"),
                (predicted, "gray", "predição"),
                (error, error_colours, "erro (vermelho FP · azul FN)"),
            ]
        ):
            axis = axes[row_index, column]
            axis.imshow(image, cmap=cmap, vmin=0, vmax=2 if column == 3 else None)
            axis.set_title(title, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        f"Conjunto de teste — {len(order)} patches de pior Dice (limiar {threshold:.2f})",
        fontsize=10,
    )
    figure.tight_layout()

    destination = config.path("figures") / "predicoes_teste_piores.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("figura escrita em %s", config.display_path(destination))

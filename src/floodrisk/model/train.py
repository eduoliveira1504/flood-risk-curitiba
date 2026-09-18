"""Estágio ``train``: treina a U-Net de segmentação de superfície impermeável.

O que a rede aprende é UMA coisa: dado um recorte de reflectância Sentinel-2,
quais pixels são superfície impermeável. Ela não prevê alagamento. O alagamento
vem depois, do cruzamento entre a impermeabilidade que sai daqui, a declividade
do terreno e o cenário de chuva — e é esse cruzamento que o ``susceptibility``
monta. Confundir os dois é o erro mais fácil de cometer ao escrever o documento.

Decisões do laço que valem registro:

- **Seleção pelo Dice de validação**, não pela perda. A perda mistura BCE e
  Dice; o documento reporta Dice. Selecionar por uma coisa e reportar outra é
  como se escolhe, sem querer, um modelo pior.
- **O conjunto de teste não é tocado.** Nem para early stopping, nem para
  escolher limiar. Ele existe para ser gasto uma vez, no ``evaluate``.
- **Precisão mista** quando há GPU. Em Blackwell isso é quase o dobro de
  velocidade sem efeito mensurável na métrica desta tarefa.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

from .. import artifacts
from ..config import Config

logger = logging.getLogger(__name__)

__all__ = ["TrainError", "build_model", "run"]


class TrainError(RuntimeError):
    """Ambiente ou insumo impedindo o treino."""


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise TrainError(
            "PyTorch não está instalado. Para a RTX 5060 (Blackwell, sm_120) a "
            "build precisa ser a de CUDA 12.8 — a do PyPI padrão não reconhece a "
            "placa e só falha na hora do treino:\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu128\n"
            '  pip install -e ".[dl]"'
        ) from exc
    return torch


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np

    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_device():
    torch = _require_torch()
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("dispositivo: cuda (%s)", name)
        return torch.device("cuda")
    logger.warning(
        "CUDA indisponível — treinando na CPU. Funciona, mas leva horas em vez "
        "de minutos. Confira a instalação do PyTorch com índice cu128."
    )
    return torch.device("cpu")


def build_model(config: Config):
    """Instancia a U-Net com o encoder configurado."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise TrainError(
            'segmentation-models-pytorch ausente. Rode: pip install -e ".[dl]"'
        ) from exc

    model = config.model
    if model.arch != "unet":
        raise TrainError(
            f"'model.arch' = {model.arch!r}; este estágio implementa apenas 'unet'"
        )

    return smp.Unet(
        encoder_name=model.encoder,
        encoder_weights=model.encoder_weights,
        in_channels=model.in_channels,
        classes=model.classes,
    )


def _loaders(config: Config, stats):
    from torch.utils.data import DataLoader

    from .data import PatchDataset, load_manifest

    train_rows = load_manifest(config, "train")
    val_rows = load_manifest(config, "val")

    common = {
        "batch_size": config.model.batch_size,
        "num_workers": config.model.num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        PatchDataset(config, train_rows, stats, augment=config.model.augment),
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        PatchDataset(config, val_rows, stats, augment=False),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, len(train_rows), len(val_rows)


def _normalisation(config: Config):
    """Estatísticas de normalização, calculadas do treino e cacheadas em disco."""
    from .data import Normalisation, compute_normalisation, load_manifest, save_normalisation

    path = artifacts.normalisation(config)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            stats = Normalisation.from_dict(json.load(handle))
        logger.info("normalização lida de %s", config.display_path(path))
        return stats

    logger.info("calculando normalização a partir do conjunto de treino")
    stats = compute_normalisation(config, load_manifest(config, "train"))
    save_normalisation(path, stats)
    logger.info("normalização escrita em %s", config.display_path(path))
    return stats


def _run_epoch(model, loader, criterion, device, threshold, optimizer=None, scaler=None):
    """Uma passada completa. Sem ``optimizer``, é avaliação."""
    torch = _require_torch()

    from .loss import confusion_counts, segmentation_metrics

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_items = 0
    tp = fp = fn = 0.0
    use_amp = scaler is not None and device.type == "cuda"

    with torch.set_grad_enabled(training):
        for image, label, valid in loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image)
                loss = criterion(logits, label, valid)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.detach()) * image.shape[0]
            total_items += image.shape[0]

            batch_tp, batch_fp, batch_fn = confusion_counts(
                logits.detach().float(), label, valid, threshold
            )
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn

    metrics = segmentation_metrics(tp, fp, fn)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def run(config: Config) -> Path:
    """Treina, salva o melhor checkpoint e o histórico por época."""
    torch = _require_torch()

    from .loss import MaskedBCEDice

    _seed_everything(config.project.random_seed)
    device = _select_device()

    stats = _normalisation(config)
    train_loader, val_loader, n_train, n_val = _loaders(config, stats)
    logger.info("treino: %d patches · validação: %d patches", n_train, n_val)

    model = build_model(config).to(device)
    criterion = MaskedBCEDice(
        bce_weight=config.model.loss.bce_weight,
        dice_weight=config.model.loss.dice_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )
    # Paciência do scheduler menor que a do early stopping, para que a taxa de
    # aprendizado ainda tenha chance de resgatar o treino antes de ele parar.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(config.model.early_stopping_patience // 3, 1),
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if config.model.amp and device.type == "cuda"
        else None
    )
    if scaler is not None:
        logger.info("precisão mista ligada")

    threshold = config.evaluation.binarization_threshold
    history: list[dict[str, float]] = []
    best_dice = -1.0
    best_epoch = 0
    checkpoint_path = artifacts.checkpoint(config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("época   perda_tr  perda_val   dice_val    iou_val    lr")
    for epoch in range(1, config.model.max_epochs + 1):
        started = time.perf_counter()
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, threshold, optimizer, scaler
        )
        val_metrics = _run_epoch(model, val_loader, criterion, device, threshold)
        scheduler.step(val_metrics["dice"])
        learning_rate = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "seconds": round(time.perf_counter() - started, 2),
                "learning_rate": learning_rate,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        logger.info(
            "%5d   %8.4f  %9.4f  %9.4f  %9.4f  %.1e%s",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["dice"],
            val_metrics["iou"],
            learning_rate,
            "  <- melhor" if val_metrics["dice"] > best_dice else "",
        )

        # Seleção pelo Dice de validação: é a métrica que o documento reporta.
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "normalisation": stats.as_dict(),
                    "architecture": {
                        "arch": config.model.arch,
                        "encoder": config.model.encoder,
                        "in_channels": config.model.in_channels,
                        "classes": config.model.classes,
                    },
                    "patch_size": config.raster.patch_size,
                    "bands": list(config.sentinel.bands),
                    "binarization_threshold": threshold,
                    "seed": config.project.random_seed,
                },
                checkpoint_path,
            )

        if epoch - best_epoch >= config.model.early_stopping_patience:
            logger.info(
                "early stopping: %d épocas sem melhora no Dice de validação",
                config.model.early_stopping_patience,
            )
            break

    _write_history(config, history)
    logger.info(
        "melhor época: %d · Dice de validação %.4f", best_epoch, best_dice
    )
    if best_dice < config.evaluation.dice_threshold:
        logger.warning(
            "Dice de validação %.4f abaixo do limiar de %.2f declarado no "
            "documento. Antes de mexer em hiperparâmetro, olhe uma predição "
            "sobreposta à imagem: o problema costuma estar no rótulo.",
            best_dice,
            config.evaluation.dice_threshold,
        )
    logger.info("checkpoint em %s", config.display_path(checkpoint_path))
    return checkpoint_path


def _write_history(config: Config, history) -> None:
    if not history:
        return
    destination = artifacts.training_history(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    logger.info("histórico escrito em %s", config.display_path(destination))

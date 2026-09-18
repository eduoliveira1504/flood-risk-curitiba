"""Perda combinada BCE + Dice, ambas mascaradas, e as métricas de segmentação.

**Por que combinar as duas.** BCE olha pixel a pixel e é bem-comportada para
otimizar, mas é indiferente à forma: errar a borda inteira de um quarteirão
custa o mesmo que errar pixels espalhados. Dice olha a sobreposição das regiões
e é a métrica que o documento reporta — mas seu gradiente é instável quando a
predição começa quase vazia. Somadas, uma dá estabilidade no começo do treino e
a outra alinha a otimização com o que será medido no fim.

**Por que tudo é mascarado.** O mosaico tem pixels sem nenhuma cena limpa. Eles
não são "área permeável": são ausência de dado. Contá-los como negativo ensina a
rede que borda de mosaico é campo aberto, e ainda por cima infla a métrica com
acertos gratuitos numa região onde não há nada para acertar.

**Sem ``pos_weight``.** A classe positiva está perto de 45% do território — o
problema é praticamente balanceado, e ponderar aqui só adicionaria um
hiperparâmetro sem base empírica. Se a prevalência mudar muito ao trocar o
recorte, esta é a primeira coisa a revisitar.

**Dice no lote inteiro, não por imagem.** Com patches de 1.280 m, alguns caem em
área totalmente permeável. Dice por imagem nesses casos é 0/0, e o remédio usual
(somar um epsilon) vira uma recompensa arbitrária que domina a média num lote
pequeno. Acumular interseção e união no lote todo evita essa indeterminação.
"""

from __future__ import annotations

__all__ = ["MaskedBCEDice", "segmentation_metrics"]


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise RuntimeError(
            "PyTorch não está instalado. Para a RTX 5060 (Blackwell, sm_120):\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu128\n"
            '  pip install -e ".[dl]"'
        ) from exc
    return torch


def _upcast(tensor):
    """Traz o tensor para float32 antes de qualquer redução.

    NÃO é zelo excessivo. Com precisão mista o logit chega em float16, e uma
    soma sobre os 262.144 pixels de um lote passa de 65.504 — o maior valor
    representável em float16. A perda viraria ``inf`` na primeira época, e o
    sintoma (NaN) aponta para a taxa de aprendizado, não para a causa. O custo de
    fazer a redução em float32 é irrelevante perto do da convolução.
    """
    return tensor.float()


def masked_bce(logits, target, valid):
    """Entropia cruzada binária média sobre os pixels válidos."""
    torch = _import_torch()
    import torch.nn.functional as functional

    logits, target, valid = _upcast(logits), _upcast(target), _upcast(valid)
    per_pixel = functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    total = valid.sum()
    if total == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return (per_pixel * valid).sum() / total


def masked_soft_dice(logits, target, valid, smooth: float = 1.0):
    """Dice contínuo (sem binarizar) acumulado no lote inteiro.

    Sem binarização porque o limiar não é diferenciável; a probabilidade entra
    direto na interseção. É o "soft Dice" padrão.
    """
    torch = _import_torch()

    logits, target, valid = _upcast(logits), _upcast(target), _upcast(valid)
    probability = torch.sigmoid(logits) * valid
    reference = target * valid

    intersection = (probability * reference).sum()
    union = probability.sum() + reference.sum()
    return 1.0 - (2.0 * intersection + smooth) / (union + smooth)


class MaskedBCEDice:
    """Soma ponderada das duas perdas, com os pesos vindos da configuração."""

    def __init__(self, bce_weight: float, dice_weight: float):
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("pesos da perda não podem ser negativos")
        if bce_weight + dice_weight == 0:
            raise ValueError("ao menos um dos pesos da perda precisa ser positivo")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def __call__(self, logits, target, valid):
        loss = 0.0
        if self.bce_weight:
            loss = loss + self.bce_weight * masked_bce(logits, target, valid)
        if self.dice_weight:
            loss = loss + self.dice_weight * masked_soft_dice(logits, target, valid)
        return loss


def segmentation_metrics(true_positive, false_positive, false_negative) -> dict[str, float]:
    """Dice, IoU, precisão e recall a partir das contagens acumuladas.

    Recebe contagens em vez de tensores porque a avaliação acumula a época
    inteira antes de dividir: calcular a métrica por lote e tirar a média
    ponderaria lotes pequenos igual aos grandes.
    """
    tp = float(true_positive)
    fp = float(false_positive)
    fn = float(false_negative)

    denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    return {
        "dice": (2 * tp / denominator) if denominator else 0.0,
        "iou": (tp / union) if union else 0.0,
        "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
    }


def confusion_counts(logits, target, valid, threshold: float):
    """Contagens de VP, FP e FN sobre os pixels válidos, já binarizadas."""
    torch = _import_torch()

    logits, target, valid = _upcast(logits), _upcast(target), _upcast(valid)
    predicted = (torch.sigmoid(logits) >= threshold).float() * valid
    reference = target * valid

    true_positive = (predicted * reference).sum().item()
    false_positive = (predicted * (1 - reference) * valid).sum().item()
    false_negative = ((1 - predicted) * reference * valid).sum().item()
    return true_positive, false_positive, false_negative

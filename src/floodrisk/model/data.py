"""Leitura dos patches e augmentation.

Três decisões que mudam o resultado, todas registradas aqui:

**1. Normalização calculada SÓ no conjunto de treino.** Média e desvio por banda
são estatísticas do dado; calculá-las sobre o conjunto inteiro faz informação da
validação e do teste vazar para dentro da entrada da rede. É um vazamento sutil,
que não aparece em nenhuma métrica e infla todas. As estatísticas ficam gravadas
no checkpoint, para que a inferência use exatamente as mesmas.

**2. Augmentation é o grupo D4 e nada além.** Oito orientações (quatro rotações
de 90° × espelhamento). Imagem orbital não tem "em cima" — um telhado girado
continua telhado, então a simetria é real e o ganho é de graça.

Deliberadamente NÃO há jitter de brilho, contraste ou cor. Reflectância de
Sentinel-2 é grandeza física calibrada: distorcer o valor de B08 ensina a rede
uma assinatura espectral que não existe na natureza, e é justamente a assinatura
espectral que separa asfalto de solo exposto. Augmentation geométrica preserva a
física; augmentation radiométrica a destrói.

**3. Máscara de validade viaja junto.** Pixel sem cena limpa não é área
permeável — é ausência de informação. Sem carregar ``valid`` até a perda, a rede
aprende que borda de mosaico é campo aberto.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "DataError",
    "Normalisation",
    "PatchDataset",
    "compute_normalisation",
    "load_manifest",
]

#: Fator de escala da reflectância de superfície do Sentinel-2 L2A.
REFLECTANCE_SCALE = 10_000.0


class DataError(RuntimeError):
    """Manifesto ausente, vazio ou incoerente com o que o treino espera."""


@dataclass(frozen=True)
class Normalisation:
    """Média e desvio por banda, em reflectância."""

    mean: list[float]
    std: list[float]

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, data) -> Normalisation:
        return cls(mean=list(data["mean"]), std=list(data["std"]))


def load_manifest(config: Config, split: str | None = None) -> list[dict[str, str]]:
    """Lê o manifesto do ``make-dataset``, opcionalmente filtrando um conjunto."""
    from .. import artifacts

    manifest = artifacts.patch_manifest(config)
    if not manifest.exists():
        raise DataError(
            f"manifesto ausente: {config.display_path(manifest)}. "
            "Rode 'make-dataset' antes."
        )

    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DataError(f"{config.display_path(manifest)} está vazio")

    if split is None:
        return rows

    selected = [row for row in rows if row["split"] == split]
    if not selected:
        available = sorted({row["split"] for row in rows})
        raise DataError(
            f"nenhum patch no conjunto '{split}'. Conjuntos no manifesto: {available}"
        )
    return selected


def _resolve(config: Config, row) -> Path:
    path = Path(row["file"])
    return path if path.is_absolute() else (config.root / path).resolve()


def compute_normalisation(config: Config, rows) -> Normalisation:
    """Média e desvio por banda sobre os pixels VÁLIDOS dos patches informados.

    Chamada apenas com o conjunto de treino. Incluir validação ou teste aqui
    seria vazamento — pequeno, invisível, e contaminando toda métrica reportada.
    """
    import numpy as np

    if not rows:
        raise DataError("sem patches para calcular a normalização")

    total = None
    total_sq = None
    count = 0

    for row in rows:
        with np.load(_resolve(config, row)) as payload:
            image = payload["image"].astype("float64") / REFLECTANCE_SCALE
            valid = payload["valid"].astype(bool)

        if not valid.any():
            continue
        selected = image[:, valid]
        if total is None:
            total = np.zeros(selected.shape[0], dtype="float64")
            total_sq = np.zeros(selected.shape[0], dtype="float64")
        total += selected.sum(axis=1)
        total_sq += (selected**2).sum(axis=1)
        count += selected.shape[1]

    if total is None or count == 0:
        raise DataError("nenhum pixel válido nos patches de treino")

    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 1e-12)
    return Normalisation(mean=mean.tolist(), std=np.sqrt(variance).tolist())


def save_normalisation(path: Path, stats: Normalisation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats.as_dict(), handle, indent=2)


class PatchDataset:
    """Patches recortados pelo ``make-dataset``, prontos para a U-Net.

    Não herda de ``torch.utils.data.Dataset`` de propósito: aquela classe é uma
    interface vazia, e o ``DataLoader`` aceita qualquer objeto com ``__len__`` e
    ``__getitem__``. Assim este módulo importa sem PyTorch instalado, e os testes
    do dataset rodam no ambiente enxuto.

    Devolve ``(imagem, rótulo, validade)``:

    - imagem ``(bandas, H, W)`` em float32, reflectância normalizada;
    - rótulo ``(1, H, W)`` em float32, 0 ou 1;
    - validade ``(1, H, W)`` em float32, 1 onde há dado.
    """

    def __init__(self, config: Config, rows, stats: Normalisation, augment: bool = False):
        import numpy as np

        if not rows:
            raise DataError("conjunto vazio")

        self._paths = [_resolve(config, row) for row in rows]
        self._ids = [row["id"] for row in rows]
        self._augment = augment
        self._mean = np.asarray(stats.mean, dtype="float32").reshape(-1, 1, 1)
        self._std = np.asarray(stats.std, dtype="float32").reshape(-1, 1, 1)
        if (self._std <= 0).any():
            raise DataError("desvio padrão nulo em alguma banda — patches idênticos?")

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def load_arrays(self, index: int):
        """Tudo que o item faz, ainda em NumPy.

        Separado do ``__getitem__`` para que a normalização e a augmentation
        possam ser testadas sem PyTorch instalado — é aqui que mora a lógica que
        pode estar errada em silêncio.
        """
        import numpy as np

        with np.load(self._paths[index]) as payload:
            image = payload["image"].astype("float32") / REFLECTANCE_SCALE
            label = payload["mask"].astype("float32")[None, ...]
            valid = payload["valid"].astype("float32")[None, ...]

        image = (image - self._mean) / self._std
        # Pixel sem dado entra na rede como zero — a média do conjunto depois da
        # normalização. Qualquer outro valor seria uma feição inventada; a perda
        # ignora esses pixels de qualquer forma, via 'valid'.
        image = np.where(valid > 0, image, 0.0).astype("float32")

        if self._augment:
            image, label, valid = _augment_d4(image, label, valid)

        return (
            np.ascontiguousarray(image, dtype="float32"),
            np.ascontiguousarray(label, dtype="float32"),
            np.ascontiguousarray(valid, dtype="float32"),
        )

    def __getitem__(self, index: int):
        import torch

        return tuple(torch.from_numpy(array) for array in self.load_arrays(index))


def _augment_d4(image, label, valid):
    """Uma das oito simetrias do quadrado, a mesma para imagem, rótulo e validade.

    Usa o gerador global do NumPy de propósito: o ``train`` o semeia uma vez, e
    assim a sequência de augmentation faz parte da semente do experimento.
    """
    import numpy as np

    turns = np.random.randint(4)
    flip = np.random.randint(2) == 1

    def apply(array):
        out = np.rot90(array, turns, axes=(-2, -1))
        return out[..., ::-1] if flip else out

    return apply(image), apply(label), apply(valid)

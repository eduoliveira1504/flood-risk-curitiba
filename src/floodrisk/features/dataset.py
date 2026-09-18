"""Recorte dos patches de treino e split espacial em blocos.

Este é o estágio que transforma dois rasters contínuos — o mosaico Sentinel-2 e
a máscara de impermeabilidade — no conjunto de amostras que a U-Net vê. Duas
decisões aqui valem mais que qualquer hiperparâmetro do treino.

**1. O split é espacial, e é espacial de verdade.**

Split aleatório de patches é o erro clássico em sensoriamento remoto: dois
patches vizinhos compartilham pixels e compartilham contexto, então um no treino
e outro no teste faz a métrica de teste medir memorização. O remédio é dividir
por blocos de território, não por amostra.

Só que dividir por bloco *não basta* se o patch for maior que o bloco, ou se o
patch puder atravessar a divisa entre um bloco de treino e um de teste — nesse
caso os mesmos pixels aparecem dos dois lados e o vazamento volta pela janela.
Por isso a regra implementada aqui é mais forte que "cada bloco vai para um
conjunto":

    um patch só é aceito se TODOS os blocos que sua pegada toca
    pertencerem ao mesmo conjunto; caso contrário ele é descartado.

Com essa regra a ausência de vazamento é demonstrável, não esperada: se um patch
de treino e um de teste se sobrepusessem, o ponto em comum estaria num bloco que
seria simultaneamente de treino e de teste — impossível por construção. O preço
é uma faixa morta em volta de cada divisa de bloco, e o estágio informa quantos
patches ela custou.

**2. Os patches cobrem a bounding box inteira, não só o município.**

O rótulo fora de Curitiba é ligeiramente pior, porque o cadastro viário do
GeoCuritiba para na divisa e ali sobra só o WorldCover. Quanto pior, exatamente?
O ``build-mask`` mediu: as vias acrescentam **0,9 ponto percentual** à classe
positiva. Degradar 0,9 p.p. do rótulo em parte da amostra é barato perto de
jogar fora metade dos patches — a bbox tem 737 km², o município tem 435 km².

O recorte municipal continua valendo para o *produto* (o mapa de suscetibilidade
sai só para Curitiba). O manifesto grava a fração de cada patch dentro do
município, para que a avaliação possa reportar a métrica restrita se a banca
pedir.
"""

from __future__ import annotations

import csv
import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

from .. import artifacts
from ..config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetError",
    "Patch",
    "SplitResult",
    "assign_blocks",
    "block_id",
    "blocks_touched",
    "build",
    "resolve_splits",
]

Block = tuple[int, int]


class DatasetError(RuntimeError):
    """Insumo ausente ou inconsistente na montagem do conjunto de treino."""


@dataclass(frozen=True)
class Patch:
    """Uma janela de recorte, com sua pegada métrica e suas estatísticas."""

    col_off: int
    row_off: int
    width: int
    height: int
    #: oeste, sul, leste, norte em ``project.crs_metric``
    bounds: tuple[float, float, float, float]
    valid_fraction: float = 1.0
    positive_fraction: float = 0.0
    aoi_fraction: float = 1.0

    @property
    def name(self) -> str:
        return f"patch_r{self.row_off:05d}_c{self.col_off:05d}"

    @property
    def centre(self) -> tuple[float, float]:
        west, south, east, north = self.bounds
        return ((west + east) / 2, (south + north) / 2)


@dataclass(frozen=True)
class SplitResult:
    kept: list[tuple[Patch, str]]
    dropped: list[Patch]
    assignment: dict[Block, str]


# --------------------------------------------------------------------------- #
# Blocos
# --------------------------------------------------------------------------- #


def block_id(x: float, y: float, block_size: float) -> Block:
    """Bloco que contém um ponto métrico.

    Piso, não truncamento: coordenada negativa acontece se o CRS mudar, e
    ``int(-0.3)`` daria 0 em vez de -1, colando dois blocos vizinhos num só.
    """
    if block_size <= 0:
        raise DatasetError("'split.block_size_m' precisa ser positivo")
    return (floor(x / block_size), floor(y / block_size))


def blocks_touched(bounds: Sequence[float], block_size: float) -> set[Block]:
    """Todos os blocos que a pegada do patch cobre com área positiva.

    O intervalo é semiaberto de propósito. Um patch cuja borda leste cai
    exatamente sobre a divisa não "toca" o bloco seguinte — divide com ele uma
    linha de área zero, e nenhum pixel. Contá-lo faria a faixa morta crescer sem
    motivo.
    """
    if block_size <= 0:
        raise DatasetError("'split.block_size_m' precisa ser positivo")

    west, south, east, north = bounds
    if east <= west or north <= south:
        raise DatasetError(f"pegada degenerada: {tuple(bounds)}")

    first_col, last_col = floor(west / block_size), ceil(east / block_size) - 1
    first_row, last_row = floor(south / block_size), ceil(north / block_size) - 1
    return {
        (col, row)
        for col in range(first_col, last_col + 1)
        for row in range(first_row, last_row + 1)
    }


def assign_blocks(
    weights: Mapping[Block, int], fractions: Mapping[str, float], seed: int
) -> dict[Block, str]:
    """Sorteia cada bloco para um conjunto, respeitando as frações pedidas.

    O sorteio é da ORDEM dos blocos; a escolha do conjunto é gulosa, sempre para
    o que está proporcionalmente mais vazio. Sortear o conjunto de cada bloco
    de forma independente daria, com poucas dezenas de blocos, desvios de dez
    pontos percentuais na divisão — e a fração é contada em patches, não em
    blocos, porque bloco de borda tem menos patch que bloco de miolo.
    """
    if not weights:
        raise DatasetError("nenhum bloco a dividir")

    names = [name for name in sorted(fractions) if fractions[name] > 0]
    if not names:
        raise DatasetError("todas as frações de 'split.fractions' são zero")

    order = sorted(weights)
    random.Random(seed).shuffle(order)

    filled = dict.fromkeys(names, 0.0)
    assignment: dict[Block, str] = {}
    for block in order:
        chosen = min(names, key=lambda name: (filled[name] / fractions[name], name))
        assignment[block] = chosen
        filled[chosen] += weights[block]
    return assignment


def resolve_splits(
    patches: Iterable[Patch],
    block_size: float,
    fractions: Mapping[str, float],
    seed: int,
) -> SplitResult:
    """Aplica o split em blocos e descarta os patches que cruzam a divisa.

    O universo de blocos inclui todo bloco TOCADO por alguma pegada, não apenas
    os que contêm um centro de patch. Um bloco de borda sem centro nenhum ainda
    pode ser coberto por dois patches de conjuntos diferentes; deixá-lo sem
    conjunto reabriria exatamente o vazamento que a regra fecha.
    """
    items = list(patches)
    if not items:
        raise DatasetError("nenhum patch candidato ao split")

    touched = [blocks_touched(patch.bounds, block_size) for patch in items]

    weights: dict[Block, int] = {block: 0 for blocks in touched for block in blocks}
    for patch in items:
        weights[block_id(*patch.centre, block_size)] += 1

    assignment = assign_blocks(weights, fractions, seed)

    kept: list[tuple[Patch, str]] = []
    dropped: list[Patch] = []
    for patch, blocks in zip(items, touched, strict=True):
        splits = {assignment[block] for block in blocks}
        if len(splits) == 1:
            kept.append((patch, splits.pop()))
        else:
            dropped.append(patch)

    return SplitResult(kept=kept, dropped=dropped, assignment=assignment)


# --------------------------------------------------------------------------- #
# Randomização restrita
# --------------------------------------------------------------------------- #


def prevalence_spread(result: SplitResult) -> float:
    """Diferença, em pontos percentuais, entre a classe positiva mais e menos
    frequente entre os conjuntos.

    É a grandeza que decide se a métrica de teste é comparável com a de
    validação. Dice sobe com a prevalência: comparar um conjunto com 42% de
    pixel impermeável contra outro com 52% mede a diferença entre os recortes de
    território, não entre modelos.
    """
    by_split: dict[str, list[float]] = {}
    for patch, split in result.kept:
        by_split.setdefault(split, []).append(patch.positive_fraction)
    if len(by_split) < 2:
        return 0.0
    means = [sum(values) / len(values) for values in by_split.values()]
    return 100 * (max(means) - min(means))


def choose_seed(
    patches: Sequence[Patch],
    block_size: float,
    fractions: Mapping[str, float],
    seed: int,
    max_spread_pp: float,
    max_attempts: int,
) -> tuple[int, SplitResult, float]:
    """Varre sementes até achar um split com conjuntos comparáveis.

    Isto NÃO é escolher o resultado que agrada. A diferença entre randomização
    restrita e garimpo de semente é que o critério é declarado na configuração
    *antes* da varredura, a busca é determinística (sementes consecutivas a
    partir de ``project.random_seed``), e a primeira semente que satisfaz o
    critério é aceita — não a melhor de todas. Qualquer pessoa reexecuta e
    encontra a mesma. É o mesmo raciocínio da randomização restrita de ensaio
    clínico: sorteia-se, mas rejeita-se o sorteio que desbalanceia a covariável
    que se sabe de antemão que contamina o desfecho.

    Quando nenhuma semente satisfaz, devolve a menos pior e cabe ao chamador
    avisar — travar o pipeline por uma propriedade estatística seria pior que
    seguir com o desvio documentado.
    """
    if max_attempts < 1:
        raise DatasetError("'split.max_seed_attempts' precisa ser pelo menos 1")

    best: tuple[int, SplitResult, float] | None = None
    for offset in range(max_attempts):
        candidate = seed + offset
        result = resolve_splits(patches, block_size, fractions, candidate)
        spread = prevalence_spread(result)

        # Semente que perde um conjunto inteiro não serve, por melhor que seja o
        # espalhamento: não dá para avaliar no que não existe.
        complete = {split for _, split in result.kept} == {
            name for name, value in fractions.items() if value > 0
        }
        if complete and spread <= max_spread_pp:
            return candidate, result, spread
        if complete and (best is None or spread < best[2]):
            best = (candidate, result, spread)

    if best is None:
        raise DatasetError(
            f"nenhuma das {max_attempts} sementes produziu todos os conjuntos de "
            "'split.fractions'. Aumente 'split.block_size_m' ou reduza o número "
            "de conjuntos."
        )
    return best


# --------------------------------------------------------------------------- #
# Estágio
# --------------------------------------------------------------------------- #


def _window_bounds(transform, window, resolution: float) -> tuple[float, float, float, float]:
    west = transform.c + window.col_off * resolution
    north = transform.f - window.row_off * resolution
    return (west, north - window.height * resolution, west + window.width * resolution, north)


def _candidates(config: Config) -> list[Patch]:
    """Percorre a grade de referência e mede cada janela candidata."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window as RasterWindow

    from ..geo import aoi_geometry, patch_windows

    image_path = artifacts.s2_mosaic(config)
    mask_path = artifacts.impervious_mask(config)
    missing = [p for p in (image_path, mask_path) if not p.exists()]
    if missing:
        names = ", ".join(config.display_path(p) for p in missing)
        raise DatasetError(
            f"insumo ausente: {names}. Rode 'acquire-sentinel' e 'build-mask' antes."
        )

    from shapely.geometry import box

    aoi = aoi_geometry(config, metric=True)

    with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask:
        if (image.width, image.height) != (mask.width, mask.height):
            raise DatasetError(
                f"imagem {image.width}x{image.height} e máscara "
                f"{mask.width}x{mask.height} não estão na mesma grade — "
                "refaça o 'build-mask'."
            )
        if image.transform != mask.transform:
            raise DatasetError(
                "imagem e máscara têm transformações diferentes; a grade de "
                "referência foi quebrada em algum estágio anterior."
            )

        resolution = config.raster.resolution_m
        patches: list[Patch] = []

        for window in patch_windows(
            image.width, image.height, config.raster.patch_size, config.raster.patch_overlap
        ):
            read = RasterWindow(window.col_off, window.row_off, window.width, window.height)
            bands = image.read(window=read)
            label = mask.read(1, window=read)

            # Pixel sem nenhuma cena limpa na janela saiu 0 em todas as bandas —
            # é assim que o evalscript e o merge marcam ausência de dado.
            valid = (bands > 0).any(axis=0)
            valid_fraction = float(valid.mean())
            if valid_fraction < config.raster.min_valid_fraction:
                continue

            bounds = _window_bounds(image.transform, window, resolution)
            footprint = box(*bounds)
            overlap = footprint.intersection(aoi)

            patches.append(
                Patch(
                    col_off=window.col_off,
                    row_off=window.row_off,
                    width=window.width,
                    height=window.height,
                    bounds=bounds,
                    valid_fraction=valid_fraction,
                    positive_fraction=float(np.count_nonzero(label) / label.size),
                    aoi_fraction=float(overlap.area / footprint.area),
                )
            )

    return patches


def _write_patch(destination: Path, image, mask, patch: Patch) -> None:
    """Grava um patch. Recebe os datasets já abertos — são centenas de patches."""
    import numpy as np
    from rasterio.windows import Window as RasterWindow

    read = RasterWindow(patch.col_off, patch.row_off, patch.width, patch.height)
    bands = image.read(window=read)
    label = mask.read(1, window=read)

    np.savez_compressed(
        destination,
        image=bands,
        mask=label,
        # Vai junto porque a perda precisa ignorar pixel sem dado: contá-lo como
        # negativo ensinaria a rede que borda de mosaico é área permeável.
        valid=(bands > 0).any(axis=0).astype("uint8"),
    )


_MANIFEST_FIELDS = (
    "id",
    "split",
    "col_off",
    "row_off",
    "width",
    "height",
    "west",
    "south",
    "east",
    "north",
    "block_col",
    "block_row",
    "valid_fraction",
    "positive_fraction",
    "aoi_fraction",
    "file",
)


def build(config: Config) -> Path:
    """Estágio ``make-dataset``: recorta os patches e grava o split."""
    import rasterio

    image_path = artifacts.s2_mosaic(config)
    mask_path = artifacts.impervious_mask(config)

    candidates = _candidates(config)
    if not candidates:
        raise DatasetError(
            "nenhum patch passou em 'raster.min_valid_fraction' "
            f"({config.raster.min_valid_fraction:.0%}) — o mosaico está vazio?"
        )

    extent_m = config.raster.patch_size * config.raster.resolution_m
    logger.info(
        "patch de %dpx (%.0f m), passo %dpx, bloco de %.0f m",
        config.raster.patch_size,
        extent_m,
        config.raster.stride,
        config.split.block_size_m,
    )
    logger.info("candidatos com dado suficiente: %d", len(candidates))

    chosen_seed, result, spread = choose_seed(
        candidates,
        block_size=config.split.block_size_m,
        fractions=config.split.fractions,
        seed=config.project.random_seed,
        max_spread_pp=config.split.max_prevalence_spread_pp,
        max_attempts=config.split.max_seed_attempts,
    )
    attempts = chosen_seed - config.project.random_seed + 1
    logger.info(
        "descartados na faixa morta entre blocos: %d (%.0f%%)",
        len(result.dropped),
        100 * len(result.dropped) / len(candidates),
    )
    if spread <= config.split.max_prevalence_spread_pp:
        logger.info(
            "semente %d (%da tentativa): espalhamento da classe positiva "
            "%.1f p.p., dentro do limite de %.1f",
            chosen_seed,
            attempts,
            spread,
            config.split.max_prevalence_spread_pp,
        )
    else:
        logger.warning(
            "nenhuma das %d sementes ficou abaixo de %.1f p.p. de espalhamento; "
            "usando a semente %d, com %.1f p.p. A métrica de teste NÃO é "
            "diretamente comparável com a de validação — reporte a prevalência "
            "de cada conjunto junto do Dice.",
            config.split.max_seed_attempts,
            config.split.max_prevalence_spread_pp,
            chosen_seed,
            spread,
        )
    if not result.kept:
        raise DatasetError(
            "a faixa morta consumiu todos os patches. Aumente "
            "'split.block_size_m' ou reduza 'raster.patch_size'."
        )

    # Limpa a corrida anterior: sobrar .npz de um split antigo é a forma mais
    # silenciosa de contaminar o teste.
    root = artifacts.patches_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    for stale in sorted(root.rglob("*.npz")):
        stale.unlink()

    rows = []
    with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask:
        for patch, split in sorted(result.kept, key=lambda item: (item[1], item[0].name)):
            folder = root / split
            folder.mkdir(parents=True, exist_ok=True)
            destination = folder / f"{patch.name}.npz"
            _write_patch(destination, image, mask, patch)

            block_col, block_row = block_id(*patch.centre, config.split.block_size_m)
            west, south, east, north = patch.bounds
            rows.append(
                {
                    "id": patch.name,
                    "split": split,
                    "col_off": patch.col_off,
                    "row_off": patch.row_off,
                    "width": patch.width,
                    "height": patch.height,
                    "west": f"{west:.1f}",
                    "south": f"{south:.1f}",
                    "east": f"{east:.1f}",
                    "north": f"{north:.1f}",
                    "block_col": block_col,
                    "block_row": block_row,
                    "valid_fraction": f"{patch.valid_fraction:.4f}",
                    "positive_fraction": f"{patch.positive_fraction:.4f}",
                    "aoi_fraction": f"{patch.aoi_fraction:.4f}",
                    "file": config.display_path(destination),
                }
            )

    manifest = artifacts.patch_manifest(config)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    _write_footprints(config, result)
    _write_split_record(config, result, chosen_seed, spread, len(candidates))
    _report(config, result)

    logger.info("manifesto escrito em %s", config.display_path(manifest))
    return manifest


def _write_split_record(
    config: Config, result: SplitResult, seed: int, spread: float, candidates: int
) -> None:
    """Registro do split em JSON — é a procedência que o documento cita.

    Sem isto, "a semente foi 47" vira folclore oral do projeto. Com isto, o
    parâmetro que governou a divisão está versionado junto do resultado.
    """
    import json

    counts: dict[str, int] = {}
    for _, split in result.kept:
        counts[split] = counts.get(split, 0) + 1

    record = {
        "strategy": config.split.strategy,
        "block_size_m": config.split.block_size_m,
        "patch_size_px": config.raster.patch_size,
        "patch_extent_m": config.raster.patch_size * config.raster.resolution_m,
        "stride_px": config.raster.stride,
        "requested_fractions": dict(config.split.fractions),
        "configured_seed": config.project.random_seed,
        "chosen_seed": seed,
        "seed_attempts": seed - config.project.random_seed + 1,
        "max_prevalence_spread_pp": config.split.max_prevalence_spread_pp,
        "prevalence_spread_pp": round(spread, 3),
        "candidates": candidates,
        "dropped_at_block_boundary": len(result.dropped),
        "kept": counts,
        "blocks": len(result.assignment),
    }
    destination = artifacts.split_record(config)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
    logger.info("registro do split escrito em %s", config.display_path(destination))


def _write_footprints(config: Config, result: SplitResult) -> None:
    """Pegadas dos patches em GeoPackage — a figura do split sai daqui."""
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {
            "id": [patch.name for patch, _ in result.kept],
            "split": [split for _, split in result.kept],
            "positive_fraction": [patch.positive_fraction for patch, _ in result.kept],
            "aoi_fraction": [patch.aoi_fraction for patch, _ in result.kept],
        },
        geometry=[box(*patch.bounds) for patch, _ in result.kept],
        crs=config.project.crs_metric,
    )
    destination = artifacts.patch_footprints(config)
    frame.to_file(destination, layer="patches", driver="GPKG")
    logger.info("pegadas escritas em %s", config.display_path(destination))


def _report(config: Config, result: SplitResult) -> None:
    """Resumo por conjunto. A coluna que importa é a de classe positiva.

    Se um conjunto tiver impermeabilidade muito diferente dos outros, o split
    pegou um recorte de cidade e outro de área rural, e a métrica de teste não
    vai ser comparável com a de validação por motivo nenhum de modelagem.
    """
    import numpy as np

    total = len(result.kept)
    logger.info(
        "%-10s %7s  %6s  %12s  %18s",
        "conjunto",
        "patches",
        "%",
        "impermeável",
        "dentro de Curitiba",
    )
    for name in sorted(config.split.fractions):
        items = [patch for patch, split in result.kept if split == name]
        if not items:
            logger.warning(
                "conjunto '%s' ficou VAZIO — aumente 'split.block_size_m' "
                "ou revise 'split.fractions'",
                name,
            )
            continue
        positive = float(np.mean([p.positive_fraction for p in items]))
        inside = float(np.mean([p.aoi_fraction for p in items]))
        logger.info(
            "%-10s %7d  %5.1f%%  %11.1f%%  %17.1f%%",
            name,
            len(items),
            100 * len(items) / total,
            100 * positive,
            100 * inside,
        )
    logger.info("%-10s %7d", "total", total)

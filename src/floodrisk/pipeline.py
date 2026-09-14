"""Registro de estágios do pipeline.

Cada etapa do projeto é um estágio nomeado e independente, que lê artefatos do
disco e escreve artefatos no disco. Nada de estado compartilhado em memória
entre estágios: o que não está materializado em ``data/`` não existe.

Estágios ainda não implementados aparecem no registro com ``run=None`` — eles
são listados pela CLI com o status ``pendente``, para que o roteiro do projeto
seja legível a partir do próprio código.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .config import Config

logger = logging.getLogger(__name__)

StageFn = Callable[[Config], None]


@dataclass(frozen=True)
class Stage:
    name: str
    phase: int
    help: str
    run: StageFn | None = None

    @property
    def implemented(self) -> bool:
        return self.run is not None


# --------------------------------------------------------------------------- #
# Fase 0 — estágios já implementados
# --------------------------------------------------------------------------- #


def _stage_bootstrap(config: Config) -> None:
    """Cria a árvore de diretórios de trabalho declarada na configuração."""
    created = config.ensure_dirs()
    if created:
        for directory in created:
            logger.info("criado: %s", config.display_path(directory))
    else:
        logger.info("todos os diretórios já existiam")


def _stage_acquire_boundary(config: Config) -> None:
    """Baixa o limite municipal oficial do IBGE."""
    from .acquisition import boundary

    boundary.acquire(config)


def _stage_probe_sentinel(config: Config) -> None:
    """Conta cenas e o tamanho do pedido sem gastar cota de download."""
    from .acquisition import sentinel

    sentinel.probe(config)


def _stage_acquire_sentinel(config: Config) -> None:
    """Baixa os tiles Sentinel-2 e costura o mosaico de mediana."""
    from .acquisition import sentinel

    sentinel.acquire(config)


def _stage_acquire_worldcover(config: Config) -> None:
    """Recorta o ESA WorldCover na grade do mosaico Sentinel-2."""
    from .acquisition import worldcover

    worldcover.acquire(config)


def _stage_acquire_forecast(config: Config) -> None:
    """Busca a previsão da Open-Meteo e grava o snapshot de fallback do site."""
    from .acquisition import forecast

    forecast.acquire(config)


def _stage_info(config: Config) -> None:
    """Imprime um resumo da configuração efetiva e do estado dos artefatos."""
    from .geo import aoi_geometry

    logger.info("projeto ......... %s", config.project.name)
    logger.info("config .......... %s", config.source_file)
    logger.info("CRS métrico ..... %s", config.project.crs_metric)
    logger.info("resolução ....... %.0f m", config.raster.resolution_m)
    logger.info(
        "patches ......... %dpx, overlap %dpx (stride %dpx)",
        config.raster.patch_size,
        config.raster.patch_overlap,
        config.raster.stride,
    )
    logger.info("grade zonal ..... %.0f m", config.grid.cell_size_m)

    try:
        aoi = aoi_geometry(config, metric=True)
        logger.info("AOI ............. %.1f km²", aoi.area / 1e6)
    except Exception as exc:  # pragma: no cover - depende de pyproj/geopandas
        logger.warning("AOI ............. indisponível (%s)", exc)

    if not config.occurrences.is_defined:
        logger.warning(
            "alvo supervisionado ainda indefinido "
            "(occurrences.source = null) — ver Fase 1"
        )

    for key, directory in config.paths.all().items():
        status = "ok" if directory.exists() else "ausente"
        logger.info("path %-14s %-8s %s", key, status, config.display_path(directory))


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #

STAGES: tuple[Stage, ...] = (
    Stage("bootstrap", 0, "Cria a árvore de diretórios de trabalho", _stage_bootstrap),
    Stage("info", 0, "Resume a configuração efetiva e o estado dos artefatos", _stage_info),
    # -- Fase 1: aquisição ------------------------------------------------- #
    Stage(
        "acquire-boundary",
        1,
        "Baixa o limite municipal oficial de Curitiba",
        _stage_acquire_boundary,
    ),
    Stage(
        "probe-sentinel",
        1,
        "Conta cenas e o tamanho do pedido, sem baixar nada",
        _stage_probe_sentinel,
    ),
    Stage(
        "acquire-sentinel",
        1,
        "Baixa e compõe o mosaico Sentinel-2 (mediana)",
        _stage_acquire_sentinel,
    ),
    Stage("acquire-osm", 1, "Baixa a malha viária do OpenStreetMap"),
    Stage(
        "acquire-worldcover",
        1,
        "Recorta o ESA WorldCover na grade de referência",
        _stage_acquire_worldcover,
    ),
    Stage("acquire-dem", 1, "Baixa e recorta o DEM base"),
    Stage("acquire-occurrences", 1, "Coleta e geocodifica ocorrências de alagamento"),
    Stage(
        "acquire-forecast",
        1,
        "Busca a previsão Open-Meteo e grava o snapshot de fallback",
        _stage_acquire_forecast,
    ),
    # -- Fase 2: preparo --------------------------------------------------- #
    Stage("build-mask", 2, "Monta a máscara de impermeabilidade (OSM ∪ WorldCover)"),
    Stage("build-terrain", 2, "Deriva declividade e acúmulo de fluxo do DEM"),
    Stage("make-dataset", 2, "Recorta patches de treino e aplica o split espacial"),
    # -- Fase 3: modelagem ------------------------------------------------- #
    Stage("train", 3, "Treina a U-Net de segmentação"),
    Stage("evaluate", 3, "Calcula Dice, IoU, matriz de confusão e Grad-CAM"),
    Stage("infer", 3, "Aplica o modelo à cidade inteira"),
    # -- Fase 4: produto --------------------------------------------------- #
    Stage("susceptibility", 4, "Compõe o índice de suscetibilidade na grade zonal"),
    Stage("report", 4, "Gera figuras e tabelas para o documento e o dashboard"),
)

STAGES_BY_NAME: dict[str, Stage] = {stage.name: stage for stage in STAGES}


class StageNotImplementedError(NotImplementedError):
    pass


def run_stage(name: str, config: Config) -> None:
    stage = STAGES_BY_NAME.get(name)
    if stage is None:
        raise KeyError(f"Estágio desconhecido: {name!r}")
    if stage.run is None:
        raise StageNotImplementedError(
            f"Estágio '{name}' pertence à Fase {stage.phase} e ainda não foi implementado."
        )
    logger.info("── estágio '%s' ──", name)
    stage.run(config)
    logger.info("── estágio '%s' concluído ──", name)

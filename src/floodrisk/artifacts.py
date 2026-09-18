"""Caminhos canônicos dos artefatos do pipeline.

Um lugar só. Nome de arquivo repetido como string literal em vários módulos é
como se perde meia hora descobrindo que um estágio grava ``s2_median.tif`` e o
seguinte procura ``s2_mosaic.tif``.

Ler esta lista de cima a baixo também descreve o fluxo de dados do projeto.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config

__all__ = [
    "boundary",
    "checkpoint",
    "dem",
    "evaluation_report",
    "impervious_mask",
    "normalisation",
    "osm_roads",
    "patch_footprints",
    "patch_manifest",
    "patches_dir",
    "roads",
    "s2_mosaic",
    "slope",
    "split_record",
    "streets",
    "training_history",
    "worldcover",
]


def boundary(config: Config) -> Path:
    """Limite municipal oficial, em WGS84. Recorta os resultados."""
    if not config.aoi.boundary_file:
        raise ValueError("'aoi.boundary_file' não está definido")
    return (config.root / config.aoi.boundary_file).resolve()


def s2_mosaic(config: Config) -> Path:
    """Mosaico Sentinel-2 de mediana. É a GRADE DE REFERÊNCIA do projeto.

    Todo raster derivado — WorldCover, máscara, declividade — é alinhado a este
    arquivo. Sem uma grade de referência única, dois rasters do mesmo AOI podem
    ficar meio pixel deslocados e o cruzamento sai enviesado sem avisar.
    """
    return config.path("data_interim") / "s2_median.tif"


def worldcover(config: Config) -> Path:
    """ESA WorldCover reamostrado para a grade de referência."""
    return config.path("data_interim") / "worldcover.tif"


def osm_roads(config: Config) -> Path:
    """Malha viária do OpenStreetMap, em CRS métrico."""
    return config.path("data_interim") / "osm_roads.gpkg"


def dem(config: Config) -> Path:
    """Modelo digital de elevação reamostrado para a grade de referência."""
    return config.path("data_interim") / "dem.tif"


def slope(config: Config) -> Path:
    """Declividade em graus, derivada do DEM. Segundo eixo do índice de risco."""
    return config.path("data_interim") / "slope.tif"


def streets(config: Config) -> Path:
    """Malha viária oficial do município (GeoCuritiba / IPPUC), em CRS métrico."""
    return config.path("data_interim") / "streets.gpkg"


def roads(config: Config) -> tuple[Path, str]:
    """Fonte viária efetiva e o nome da sua camada, conforme a configuração.

    Existe para que o ``build-mask`` não precise saber de onde as vias vieram —
    trocar de fonte é mexer no YAML, não no código da máscara.
    """
    if config.ground_truth.roads_source == "osm":
        return osm_roads(config), "roads"
    return streets(config), "streets"


def impervious_mask(config: Config) -> Path:
    """Máscara binária de impermeabilidade — o rótulo da U-Net."""
    return config.path("data_processed") / "impervious_mask.tif"


def patches_dir(config: Config) -> Path:
    """Raiz dos patches de treino, com uma subpasta por conjunto do split."""
    return config.path("data_processed") / "patches"


def patch_manifest(config: Config) -> Path:
    """Manifesto dos patches: split, pegada, bloco e estatísticas de cada um.

    É por ele que o treino carrega o conjunto, e é ele que documenta o split
    para quem for reproduzir o experimento.
    """
    return patches_dir(config) / "manifest.csv"


def patch_footprints(config: Config) -> Path:
    """Pegadas dos patches em GeoPackage — insumo da figura do split espacial."""
    return patches_dir(config) / "patches.gpkg"


def normalisation(config: Config) -> Path:
    """Média e desvio por banda, calculados SÓ no conjunto de treino.

    Fica ao lado dos patches, não junto do modelo, porque é propriedade do
    conjunto de dados: trocar de arquitetura não muda estas estatísticas.
    """
    return patches_dir(config) / "normalisation.json"


def checkpoint(config: Config) -> Path:
    """Melhor checkpoint da U-Net, selecionado pelo Dice de validação."""
    return config.path("models") / "unet_best.pt"


def evaluation_report(config: Config) -> Path:
    """Métricas do conjunto de teste. Escrito uma vez, pelo estágio 'evaluate'."""
    return config.path("reports") / "evaluation.json"


def training_history(config: Config) -> Path:
    """Métrica por época — a curva de treino do documento sai daqui."""
    return config.path("models") / "training_history.csv"


def split_record(config: Config) -> Path:
    """Procedência do split: semente efetiva, critério e contagens resultantes.

    O documento cita este arquivo. Parâmetro de divisão que só existe no log de
    uma execução não é reprodutível por terceiro.
    """
    return patches_dir(config) / "split.json"

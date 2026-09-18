"""Carga e validação da configuração central do pipeline.

O YAML em ``configs/default.yaml`` é a única fonte de verdade dos parâmetros.
Aqui ele vira um grafo de dataclasses tipadas, com validação estrita: qualquer
chave desconhecida ou ausente derruba a carga imediatamente, em vez de virar um
``KeyError`` três horas depois no meio de um treino.
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

_ROOT_MARKERS = ("pyproject.toml", ".git")

# CRS aceitos pelo Process API do Sentinel Hub: WGS84, Pseudo-Mercator e as
# zonas UTM WGS84 — 32601–32660 (norte) e 32701–32760 (sul). As zonas vão de
# 01 a 60; 32600 e 32661 não existem.
_SUPPORTED_REQUEST_CRS = re.compile(r"EPSG:(4326|3857|32[67](0[1-9]|[1-5]\d|60))")


def find_repo_root(start: Path | None = None) -> Path:
    """Sobe a árvore de diretórios até achar a raiz do repositório."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise RuntimeError(
        "Raiz do repositório não encontrada (procurando por pyproject.toml ou .git)."
    )


class ConfigError(ValueError):
    """Erro de estrutura, tipo ou valor na configuração."""


def _build(cls: type[T], data: Any, path: str) -> T:
    """Instancia uma dataclass a partir de um mapping, rejeitando chaves extras."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"'{path}': esperado um mapeamento, recebido {type(data).__name__}")

    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"'{path}': chave(s) desconhecida(s): {sorted(unknown)}")

    required = {
        f.name
        for f in dataclasses.fields(cls)  # type: ignore[arg-type]
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    }
    missing = required - set(data)
    if missing:
        raise ConfigError(f"'{path}': chave(s) obrigatória(s) ausente(s): {sorted(missing)}")

    return cls(**dict(data))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Seções
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    crs_geo: str
    crs_metric: str
    random_seed: int


@dataclass(frozen=True)
class AOIConfig:
    bbox: list[float]
    boundary_file: str | None = None

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ConfigError("'aoi.bbox': esperado [oeste, sul, leste, norte]")
        west, south, east, north = self.bbox
        if west >= east or south >= north:
            raise ConfigError(f"'aoi.bbox': degenerada ou invertida: {self.bbox}")


@dataclass(frozen=True)
class BoundaryConfig:
    """Limite municipal oficial — define a área de estudo real.

    A bbox da configuração serve para aquisição; é este polígono que recorta os
    resultados. Em Curitiba a diferença é de quase 70% de área.
    """

    source: str
    url: str
    municipality_code: str
    source_crs: str
    expected_area_km2: float
    area_tolerance_pct: float
    request_timeout_s: float
    max_retries: int

    def __post_init__(self) -> None:
        if self.expected_area_km2 <= 0:
            raise ConfigError("'boundary.expected_area_km2' precisa ser positivo")
        if not 0 < self.area_tolerance_pct < 100:
            raise ConfigError("'boundary.area_tolerance_pct' precisa estar em (0, 100)")
        if not self.municipality_code.isdigit():
            raise ConfigError(
                f"'boundary.municipality_code' = {self.municipality_code!r} não é "
                "numérico; o código IBGE de Curitiba é 4106902"
            )

    @property
    def area_range_km2(self) -> tuple[float, float]:
        margin = self.expected_area_km2 * self.area_tolerance_pct / 100.0
        return (self.expected_area_km2 - margin, self.expected_area_km2 + margin)


@dataclass(frozen=True)
class RasterConfig:
    resolution_m: float
    patch_size: int
    patch_overlap: int
    min_valid_fraction: float
    inference_overlap: int

    def __post_init__(self) -> None:
        if self.patch_overlap >= self.patch_size:
            raise ConfigError("'raster.patch_overlap' precisa ser menor que 'patch_size'")
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ConfigError("'raster.min_valid_fraction' precisa estar em [0, 1]")
        if not 0 <= self.inference_overlap < self.patch_size:
            raise ConfigError(
                "'raster.inference_overlap' precisa estar em [0, patch_size) — "
                "sobreposição igual ao patch faria a janela não avançar."
            )
        if self.inference_overlap < self.patch_overlap:
            raise ConfigError(
                "'raster.inference_overlap' menor que 'patch_overlap' não faz "
                "sentido: a inferência precisa de MAIS sobreposição que o treino "
                "para que a média ponderada apague a costura entre janelas."
            )

    @property
    def stride(self) -> int:
        """Passo efetivo do janelamento de patches."""
        return self.patch_size - self.patch_overlap


@dataclass(frozen=True)
class GridConfig:
    cell_size_m: float
    min_valid_fraction: float

    def __post_init__(self) -> None:
        if self.cell_size_m <= 0:
            raise ConfigError("'grid.cell_size_m' precisa ser positivo")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ConfigError("'grid.min_valid_fraction' precisa estar em (0, 1]")


@dataclass(frozen=True)
class SentinelConfig:
    """Sentinel-2 via Copernicus Data Space Ecosystem (CDSE).

    CDSE é o serviço gratuito da ESA, não o Sentinel Hub comercial. Os endpoints
    ficam aqui, versionados, porque errar o host é o modo mais comum de gastar
    uma tarde com HTTP 401.
    """

    collection: str
    base_url: str
    token_url: str
    request_crs: str
    date_start: str
    date_end: str
    max_cloud_cover: int
    bands: list[str]
    composite: str
    request_tile_px: int
    invalid_scl_classes: list[int]
    max_processing_units: float

    def __post_init__(self) -> None:
        if "apps.sentinel-hub.com" in self.base_url:
            raise ConfigError(
                "'sentinel.base_url' aponta para o Sentinel Hub comercial. "
                "Use o CDSE: https://sh.dataspace.copernicus.eu"
            )
        # O Process API aceita só WGS84, Pseudo-Mercator e as zonas UTM WGS84.
        # SIRGAS 2000 (31982) é recusado com "Unsupported CRS value", por isso a
        # aquisição usa um CRS próprio e a reprojeção para o CRS de análise
        # acontece depois, no mosaico.
        if not _SUPPORTED_REQUEST_CRS.fullmatch(self.request_crs):
            raise ConfigError(
                f"'sentinel.request_crs' = {self.request_crs!r} não é aceito pelo "
                "Process API. Use EPSG:4326, EPSG:3857 ou uma zona UTM WGS84 "
                "(EPSG:326xx / EPSG:327xx). Para Curitiba: EPSG:32722."
            )
        if not 0 <= self.max_cloud_cover <= 100:
            raise ConfigError("'sentinel.max_cloud_cover' precisa estar em [0, 100]")
        if self.date_start >= self.date_end:
            raise ConfigError(
                f"'sentinel': date_start ({self.date_start}) não é anterior a "
                f"date_end ({self.date_end})"
            )
        if not self.bands:
            raise ConfigError("'sentinel.bands' não pode ser vazio")
        if self.composite != "median":
            raise ConfigError(
                "'sentinel.composite': apenas 'median' é implementado — é o que o "
                "documento do TCC declara."
            )
        # O Process API recusa saída acima de 2500 px por lado.
        if not 256 <= self.request_tile_px <= 2500:
            raise ConfigError("'sentinel.request_tile_px' precisa estar em [256, 2500]")
        if self.max_processing_units <= 0:
            raise ConfigError("'sentinel.max_processing_units' precisa ser positivo")
        if any(not 0 <= c <= 11 for c in self.invalid_scl_classes):
            raise ConfigError(
                "'sentinel.invalid_scl_classes': a Scene Classification Layer do "
                "Sentinel-2 só tem classes de 0 a 11"
            )


@dataclass(frozen=True)
class WorldCoverConfig:
    year: int
    version: str
    builtup_class: int
    url_template: str
    tile_size_deg: int

    def __post_init__(self) -> None:
        required = {"{version}", "{year}", "{tile}"}
        missing = {token for token in required if token not in self.url_template}
        if missing:
            raise ConfigError(
                f"'ground_truth.worldcover.url_template' sem os marcadores "
                f"{sorted(missing)}"
            )
        if self.tile_size_deg < 1:
            raise ConfigError("'ground_truth.worldcover.tile_size_deg' precisa ser >= 1")


@dataclass(frozen=True)
class OSMConfig:
    road_buffers_m: dict[str, float]
    overpass_urls: list[str]
    request_timeout_s: float
    max_retries: int

    def __post_init__(self) -> None:
        if not self.overpass_urls:
            raise ConfigError(
                "'ground_truth.osm.overpass_urls' precisa listar ao menos um espelho"
            )
        if not self.road_buffers_m:
            raise ConfigError("'ground_truth.osm.road_buffers_m' não pode ser vazio")
        invalid = {k: v for k, v in self.road_buffers_m.items() if v <= 0}
        if invalid:
            raise ConfigError(
                f"'ground_truth.osm.road_buffers_m': buffer não positivo em {sorted(invalid)}"
            )


@dataclass(frozen=True)
class StreetsConfig:
    """Malha viária oficial do município, via GeoCuritiba / IPPUC (ArcGIS REST)."""

    source: str
    service_url: str
    layer_id: int
    layer_name: str
    source_crs: str
    out_fields: list[str]
    hierarchy_field: str
    order_by_field: str
    page_size: int
    max_pages: int
    request_timeout_s: float
    max_retries: int
    default_buffer_m: float
    hierarchy_buffers_m: dict[str, float]

    def __post_init__(self) -> None:
        if self.source != "geocuritiba":
            raise ConfigError(
                "'ground_truth.streets.source': apenas 'geocuritiba' é implementado"
            )
        if self.default_buffer_m <= 0:
            raise ConfigError("'ground_truth.streets.default_buffer_m' precisa ser > 0")
        invalid = {k: v for k, v in self.hierarchy_buffers_m.items() if v <= 0}
        if invalid:
            raise ConfigError(
                f"'ground_truth.streets.hierarchy_buffers_m': buffer não positivo "
                f"em {sorted(invalid)}"
            )
        # O serviço limita a 2000 registros por página; pedir mais é ignorado em
        # silêncio e a paginação passa a pular feições.
        if not 1 <= self.page_size <= 2000:
            raise ConfigError(
                "'ground_truth.streets.page_size' precisa estar em [1, 2000]"
            )
        if self.hierarchy_field not in self.out_fields:
            raise ConfigError(
                f"'ground_truth.streets.hierarchy_field' ({self.hierarchy_field}) "
                "precisa estar em 'out_fields', senão o atributo não é baixado"
            )
        if self.order_by_field not in self.out_fields:
            raise ConfigError(
                f"'ground_truth.streets.order_by_field' ({self.order_by_field}) "
                "precisa estar em 'out_fields' para a paginação ser estável"
            )


@dataclass(frozen=True)
class GroundTruthConfig:
    roads_source: str
    worldcover: WorldCoverConfig
    osm: OSMConfig
    streets: StreetsConfig

    def __post_init__(self) -> None:
        if self.roads_source not in {"geocuritiba", "osm"}:
            raise ConfigError(
                "'ground_truth.roads_source': esperado 'geocuritiba' ou 'osm', "
                f"veio {self.roads_source!r}"
            )


@dataclass(frozen=True)
class TerrainConfig:
    dem_source: str
    url_template: str
    slope_units: str
    resample_to_resolution: bool
    expected_elevation_range_m: list[float]

    def __post_init__(self) -> None:
        if "{tile}" not in self.url_template:
            raise ConfigError("'terrain.url_template' precisa conter o marcador {tile}")
        if self.slope_units != "degrees":
            raise ConfigError("'terrain.slope_units': apenas 'degrees' é implementado")
        if len(self.expected_elevation_range_m) != 2:
            raise ConfigError(
                "'terrain.expected_elevation_range_m': esperado [mínimo, máximo]"
            )
        low, high = self.expected_elevation_range_m
        if low >= high:
            raise ConfigError(
                f"'terrain.expected_elevation_range_m': {low} não é menor que {high}"
            )


@dataclass(frozen=True)
class OccurrencesConfig:
    """Alvo supervisionado. ``source`` nulo enquanto a fonte não for definida."""

    source: str | None
    path: str
    date_field: str
    positional_tolerance_m: float

    @property
    def is_defined(self) -> bool:
        return self.source is not None


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    block_size_m: float
    fractions: dict[str, float]
    max_prevalence_spread_pp: float
    max_seed_attempts: int

    def __post_init__(self) -> None:
        total = sum(self.fractions.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"'split.fractions' deve somar 1.0, somou {total}")
        if self.strategy != "spatial_block":
            raise ConfigError(
                "'split.strategy': apenas 'spatial_block' é aceito — splits aleatórios "
                "ou cronológicos vazam contexto espacial entre tiles vizinhos."
            )
        if self.max_prevalence_spread_pp <= 0:
            raise ConfigError(
                "'split.max_prevalence_spread_pp' precisa ser positivo — é o "
                "limite de desbalanceamento aceito entre os conjuntos."
            )
        if self.max_seed_attempts < 1:
            raise ConfigError("'split.max_seed_attempts' precisa ser pelo menos 1")


@dataclass(frozen=True)
class LossConfig:
    bce_weight: float
    dice_weight: float


@dataclass(frozen=True)
class ModelConfig:
    arch: str
    encoder: str
    encoder_weights: str | None
    in_channels: int
    classes: int
    loss: LossConfig
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    scheduler: str
    num_workers: int
    amp: bool
    augment: bool

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ConfigError("'model.in_channels' precisa ser pelo menos 1")
        if self.batch_size < 1:
            raise ConfigError("'model.batch_size' precisa ser pelo menos 1")
        if self.max_epochs < 1:
            raise ConfigError("'model.max_epochs' precisa ser pelo menos 1")
        if self.early_stopping_patience < 1:
            raise ConfigError("'model.early_stopping_patience' precisa ser pelo menos 1")
        if self.num_workers < 0:
            raise ConfigError("'model.num_workers' não pode ser negativo")


@dataclass(frozen=True)
class ReferencePoint:
    """Ponto único usado para a série de chuva da cidade.

    A previsão entra como variável city-wide: o modelo de reanálise tem célula de
    9–11 km, então não há informação intraurbana a extrair. O que varia no mapa é
    o território, não a chuva.
    """

    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ConfigError(f"'forecast.reference_point.lat' fora de faixa: {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ConfigError(f"'forecast.reference_point.lon' fora de faixa: {self.lon}")


@dataclass(frozen=True)
class ForecastConfig:
    provider: str
    forecast_endpoint: str
    archive_endpoint: str
    timezone: str
    horizon_hours: int
    reference_point: ReferencePoint
    accumulation_window_h: int
    attribution: str
    request_timeout_s: float
    max_retries: int

    def __post_init__(self) -> None:
        if self.provider != "open-meteo":
            raise ConfigError("'forecast.provider': apenas 'open-meteo' é implementado")
        if not 1 <= self.horizon_hours <= 384:  # Open-Meteo entrega até 16 dias
            raise ConfigError("'forecast.horizon_hours' precisa estar em [1, 384]")
        if self.accumulation_window_h < 1:
            raise ConfigError("'forecast.accumulation_window_h' precisa ser >= 1")
        if self.accumulation_window_h > self.horizon_hours:
            raise ConfigError(
                "'forecast.accumulation_window_h' não pode exceder 'horizon_hours'"
            )
        if not self.attribution.strip():
            raise ConfigError(
                "'forecast.attribution' não pode ser vazio — a Open-Meteo é CC-BY 4.0 "
                "e exige crédito visível na página."
            )


@dataclass(frozen=True)
class ScenarioTier:
    """Patamar de chuva que seleciona uma camada de risco pré-materializada.

    Dois limiares, como nos critérios do INMET, porque alagamento urbano
    responde a intensidade e não só a volume: 30 mm em 1 h alaga, os mesmos
    30 mm espalhados por 24 h não. Um número só não distingue os dois casos.
    """

    name: str
    label: str
    hourly_mm: float
    daily_mm: float
    inmet_alert: str

    def __post_init__(self) -> None:
        for field_name in ("hourly_mm", "daily_mm"):
            if getattr(self, field_name) < 0:
                raise ConfigError(
                    f"'risk_scenarios' tier '{self.name}': {field_name} negativo"
                )


#: Janelas de acumulação dos critérios do INMET. Não são parâmetro livre — são
#: a definição do critério, então ficam no código e não no YAML.
INMET_HOURLY_WINDOW_H = 1
INMET_DAILY_WINDOW_H = 24


@dataclass(frozen=True)
class RiskScenariosConfig:
    tiers: list[ScenarioTier]
    justification_pending: bool
    source: str

    def __post_init__(self) -> None:
        if len(self.tiers) < 2:
            raise ConfigError("'risk_scenarios.tiers': precisa de ao menos dois patamares")
        if not self.source.strip():
            raise ConfigError(
                "'risk_scenarios.source': informe a procedência dos patamares. "
                "Corte de chuva sem fonte citável é o que a banca vai perguntar."
            )
        first = self.tiers[0]
        if first.hourly_mm != 0.0 or first.daily_mm != 0.0:
            raise ConfigError(
                "'risk_scenarios.tiers': o primeiro patamar precisa começar em 0 mm "
                "nos dois critérios, senão existe chuva sem cenário atribuído."
            )
        for label in ("hourly_mm", "daily_mm"):
            values = [getattr(tier, label) for tier in self.tiers]
            if values != sorted(values) or len(set(values)) != len(values):
                raise ConfigError(
                    f"'risk_scenarios.tiers': {label} precisa ser estritamente "
                    "crescente — senão a classificação por esse critério é ambígua."
                )

    def classify(self, hourly_mm: float, daily_mm: float) -> ScenarioTier:
        """Patamar correspondente, pelo critério mais severo dos dois.

        Vale o maior patamar atingido por QUALQUER um dos critérios, que é como
        o INMET opera o "ou" dos seus avisos. Como os dois limiares crescem
        junto com o patamar, basta guardar o último que passou.
        """
        if hourly_mm < 0 or daily_mm < 0:
            raise ValueError("acumulado não pode ser negativo")
        chosen = self.tiers[0]
        for tier in self.tiers:
            if hourly_mm >= tier.hourly_mm or daily_mm >= tier.daily_mm:
                chosen = tier
        return chosen


@dataclass(frozen=True)
class EvaluationConfig:
    dice_threshold: float
    binarization_threshold: float


@dataclass(frozen=True)
class TrackingConfig:
    mlflow_uri: str
    experiment_name: str


@dataclass(frozen=True)
class PathsConfig:
    """Caminhos relativos à raiz do repositório, resolvidos em absolutos."""

    data_raw: str
    data_interim: str
    data_processed: str
    models: str
    reports: str
    figures: str
    logs: str

    _root: Path = field(default=Path("."), repr=False, compare=False)

    def resolve(self, key: str) -> Path:
        value = getattr(self, key)
        if not isinstance(value, str):
            raise ConfigError(f"'paths.{key}' não é um caminho")
        return (self._root / value).resolve()

    def all(self) -> dict[str, Path]:
        return {
            f.name: self.resolve(f.name)
            for f in dataclasses.fields(self)
            if not f.name.startswith("_")
        }


# --------------------------------------------------------------------------- #
# Raiz
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    aoi: AOIConfig
    boundary: BoundaryConfig
    raster: RasterConfig
    grid: GridConfig
    sentinel: SentinelConfig
    ground_truth: GroundTruthConfig
    terrain: TerrainConfig
    occurrences: OccurrencesConfig
    forecast: ForecastConfig
    risk_scenarios: RiskScenariosConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig
    paths: PathsConfig

    root: Path = field(default=Path("."), compare=False)
    source_file: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # Coerência entre seções. Cada dataclass valida o que está dentro dela;
        # o que depende de duas seções só pode ser checado aqui.
        patch_extent_m = self.raster.patch_size * self.raster.resolution_m
        if self.split.block_size_m < 2 * patch_extent_m:
            raise ConfigError(
                f"'split.block_size_m' ({self.split.block_size_m:.0f} m) precisa "
                f"ser pelo menos o dobro da pegada do patch "
                f"({patch_extent_m:.0f} m = raster.patch_size × resolution_m). "
                "O split espacial descarta todo patch que cruza a divisa entre "
                "blocos de conjuntos diferentes; com bloco menor que isso, quase "
                "nenhum patch sobrevive."
            )

    # -- construção ------------------------------------------------------- #

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], root: Path, source: Path | None = None) -> Config:
        expected = {"project", "aoi", "boundary", "raster", "grid", "sentinel", "ground_truth",
                    "terrain", "occurrences", "forecast", "risk_scenarios", "split",
                    "model", "evaluation", "tracking", "paths"}
        unknown = set(data) - expected
        if unknown:
            raise ConfigError(f"Seção(ões) desconhecida(s) no YAML: {sorted(unknown)}")
        missing = expected - set(data)
        if missing:
            raise ConfigError(f"Seção(ões) ausente(s) no YAML: {sorted(missing)}")

        gt_raw = data["ground_truth"]
        ground_truth = GroundTruthConfig(
            roads_source=gt_raw.get("roads_source", "geocuritiba"),
            worldcover=_build(
                WorldCoverConfig, gt_raw.get("worldcover"), "ground_truth.worldcover"
            ),
            osm=_build(OSMConfig, gt_raw.get("osm"), "ground_truth.osm"),
            streets=_build(StreetsConfig, gt_raw.get("streets"), "ground_truth.streets"),
        )

        model_raw = dict(data["model"])
        model_raw["loss"] = _build(LossConfig, model_raw.get("loss"), "model.loss")

        forecast_raw = dict(data["forecast"])
        forecast_raw["reference_point"] = _build(
            ReferencePoint, forecast_raw.get("reference_point"), "forecast.reference_point"
        )

        scenarios_raw = dict(data["risk_scenarios"])
        tiers_raw = scenarios_raw.get("tiers")
        if not isinstance(tiers_raw, list):
            raise ConfigError("'risk_scenarios.tiers': esperado uma lista")
        scenarios_raw["tiers"] = [
            _build(ScenarioTier, tier, f"risk_scenarios.tiers[{index}]")
            for index, tier in enumerate(tiers_raw)
        ]

        paths_raw = dict(data["paths"])
        paths_raw["_root"] = root

        return cls(
            project=_build(ProjectConfig, data["project"], "project"),
            aoi=_build(AOIConfig, data["aoi"], "aoi"),
            boundary=_build(BoundaryConfig, data["boundary"], "boundary"),
            raster=_build(RasterConfig, data["raster"], "raster"),
            grid=_build(GridConfig, data["grid"], "grid"),
            sentinel=_build(SentinelConfig, data["sentinel"], "sentinel"),
            ground_truth=ground_truth,
            terrain=_build(TerrainConfig, data["terrain"], "terrain"),
            occurrences=_build(OccurrencesConfig, data["occurrences"], "occurrences"),
            forecast=_build(ForecastConfig, forecast_raw, "forecast"),
            risk_scenarios=_build(RiskScenariosConfig, scenarios_raw, "risk_scenarios"),
            split=_build(SplitConfig, data["split"], "split"),
            model=_build(ModelConfig, model_raw, "model"),
            evaluation=_build(EvaluationConfig, data["evaluation"], "evaluation"),
            tracking=_build(TrackingConfig, data["tracking"], "tracking"),
            paths=_build(PathsConfig, paths_raw, "paths"),
            root=root,
            source_file=source,
        )

    # -- conveniências ----------------------------------------------------- #

    def path(self, key: str) -> Path:
        return self.paths.resolve(key)

    def display_path(self, path: Path) -> str:
        """Caminho relativo à raiz quando possível, absoluto quando não.

        ``Path.relative_to`` levanta ``ValueError`` para qualquer caminho fora da
        raiz. Usá-lo direto numa linha de log faz o estágio morrer DEPOIS de
        concluir o trabalho, só para imprimir — o que é a pior hora possível.
        """
        try:
            return str(Path(path).relative_to(self.root))
        except ValueError:
            return str(path)

    def ensure_dirs(self) -> list[Path]:
        created = []
        for directory in self.paths.all().values():
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory)
        return created


_DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | os.PathLike[str] | None = None, root: Path | None = None) -> Config:
    """Carrega a configuração.

    Precedência: argumento explícito > ``FLOODRISK_CONFIG`` > ``configs/default.yaml``.
    """
    repo_root = root or find_repo_root()
    chosen = path or os.environ.get("FLOODRISK_CONFIG") or (repo_root / _DEFAULT_CONFIG)
    config_path = Path(chosen)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    if not config_path.exists():
        raise ConfigError(f"Arquivo de configuração não encontrado: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path} não contém um mapeamento YAML no topo")

    return Config.from_dict(raw, root=repo_root, source=config_path)

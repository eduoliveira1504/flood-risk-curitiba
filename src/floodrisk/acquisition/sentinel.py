"""Aquisição de Sentinel-2 L2A pelo Copernicus Data Space Ecosystem.

Estratégia, e por quê:

1. **Mediana calculada no servidor.** O evalscript é multi-temporal
   (``mosaicking: "ORBIT"``) e reduz a série a uma mediana por banda. A
   alternativa — baixar cada data e reduzir localmente — custaria uma requisição
   por cena e queimaria a cota gratuita (10.000 requisições/mês) sem necessidade.
2. **Máscara de nuvem antes da mediana.** Cada amostra passa pela Scene
   Classification Layer; pixel de nuvem, sombra, cirrus ou neve é descartado da
   lista antes de ordenar. Mediana sobre pixel contaminado não é composição, é
   média de erro.
3. **Requisição em tiles.** O Process API recusa saída acima de 2500 px por lado
   e Curitiba tem ~2100 × 3400 px a 10 m. A cidade é pedida em pedaços e
   costurada com rasterio.
4. **Um GeoTIFF por tile em ``data/raw/``, mosaico em ``data/interim/``.** O
   bruto nunca é sobrescrito: se a costura mudar, ela é refeita do disco, sem
   gastar cota de novo.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import Config, SentinelConfig
from ..geo import snap_bounds

logger = logging.getLogger(__name__)

__all__ = [
    "SentinelError",
    "TileRequest",
    "acquire",
    "build_evalscript",
    "build_sh_config",
    "credentials_from_env",
    "probe",
    "request_tiles",
]

CLIENT_ID_ENV = "CDSE_CLIENT_ID"
CLIENT_SECRET_ENV = "CDSE_CLIENT_SECRET"

# Identificadores das respostas do Process API.
RESPONSE_BANDS = "bands"
RESPONSE_MASK = "mask"

# L2A em DN: reflectância × 10000. Registrado no GeoTIFF para quem ler depois.
REFLECTANCE_SCALE = 10_000

# Unidade de cobrança do Sentinel Hub: uma requisição de 512 × 512 px, 3 bandas
# de entrada, 1 amostra por pixel, saída de 8 ou 16 bits = 1 PU.
PU_BASE_PIXELS = 512 * 512
PU_BASE_BANDS = 3
PU_MIN_SIZE_FACTOR = 0.01

# Além das bandas pedidas, o evalscript sempre consome SCL e dataMask.
EXTRA_INPUT_BANDS = 2


class SentinelError(RuntimeError):
    """Falha de credencial, de configuração ou de download do CDSE."""


@dataclass(frozen=True)
class TileRequest:
    """Um retângulo de requisição, em CRS métrico."""

    index: int
    bounds: tuple[float, float, float, float]
    width_px: int
    height_px: int

    @property
    def pixels(self) -> int:
        return self.width_px * self.height_px

    @property
    def name(self) -> str:
        return f"tile_{self.index:03d}"


# --------------------------------------------------------------------------- #
# Evalscript
# --------------------------------------------------------------------------- #


def build_evalscript(bands: Sequence[str], invalid_scl_classes: Sequence[int]) -> str:
    """Gera o evalscript multi-temporal de mediana para as bandas pedidas.

    Gerado a partir da configuração em vez de escrito à mão: mudar
    ``sentinel.bands`` no YAML não pode exigir editar JavaScript.
    """
    if not bands:
        raise SentinelError("nenhuma banda pedida")

    band_list = list(bands)
    inputs = json.dumps([*band_list, "SCL", "dataMask"])
    invalid = json.dumps(list(invalid_scl_classes))

    # Um acumulador por banda, preenchido só com amostras válidas.
    declarations = "\n".join(f"  var acc_{b} = [];" for b in band_list)
    pushes = "\n".join(f"    acc_{b}.push(sample.{b});" for b in band_list)
    medians = ", ".join(f"median(acc_{b})" for b in band_list)
    first = band_list[0]

    return f"""//VERSION=3
// Composição por mediana temporal com descarte de nuvem via Scene Classification
// Layer. Gerado por floodrisk.acquisition.sentinel — não edite à mão.

function setup() {{
  return {{
    input: [{{ bands: {inputs}, units: "DN" }}],
    output: [
      {{ id: "{RESPONSE_BANDS}", bands: {len(band_list)}, sampleType: "UINT16" }},
      {{ id: "{RESPONSE_MASK}", bands: 1, sampleType: "UINT8" }}
    ],
    mosaicking: "ORBIT"
  }};
}}

var INVALID_SCL = {invalid};

function isValid(sample) {{
  if (sample.dataMask !== 1) return false;
  return INVALID_SCL.indexOf(sample.SCL) === -1;
}}

function median(values) {{
  var n = values.length;
  if (n === 0) return 0;
  values.sort(function (a, b) {{ return a - b; }});
  var mid = Math.floor(n / 2);
  if (n % 2 === 1) return values[mid];
  return Math.round((values[mid - 1] + values[mid]) / 2);
}}

function evaluatePixel(samples) {{
{declarations}

  for (var i = 0; i < samples.length; i++) {{
    var sample = samples[i];
    if (!isValid(sample)) continue;
{pushes}
  }}

  // A máscara diz quantas observações sobraram: 0 significa que o pixel ficou
  // sem nenhuma cena limpa na janela e não deve entrar no treino.
  var observations = acc_{first}.length;

  return {{
    {RESPONSE_BANDS}: [{medians}],
    {RESPONSE_MASK}: [observations > 0 ? 1 : 0]
  }};
}}
"""


# --------------------------------------------------------------------------- #
# Tiles
# --------------------------------------------------------------------------- #


def request_tiles(
    bounds: Sequence[float], resolution_m: float, tile_px: int
) -> list[TileRequest]:
    """Divide o envelope em tiles de no máximo ``tile_px`` pixels por lado.

    Os limites são alinhados à resolução antes de dividir, para que os tiles
    encaixem sem meio pixel de folga e a costura não produza costura visível.
    """
    if tile_px < 1:
        raise SentinelError("tile_px precisa ser positivo")
    if resolution_m <= 0:
        raise SentinelError("resolution_m precisa ser positiva")

    west, south, east, north = snap_bounds(bounds, resolution_m)
    tile_span = tile_px * resolution_m

    tiles: list[TileRequest] = []
    index = 0
    y = south
    while y < north:
        top = min(y + tile_span, north)
        x = west
        while x < east:
            right = min(x + tile_span, east)
            width = round((right - x) / resolution_m)
            height = round((top - y) / resolution_m)
            if width > 0 and height > 0:
                tiles.append(
                    TileRequest(
                        index=index,
                        bounds=(x, y, right, top),
                        width_px=width,
                        height_px=height,
                    )
                )
                index += 1
            x = right
        y = top
    return tiles


# --------------------------------------------------------------------------- #
# Custo
# --------------------------------------------------------------------------- #


def estimate_processing_units(
    output_pixels: int, n_bands: int, n_samples_per_pixel: int
) -> float:
    """Estima o custo em processing units de um pedido multi-temporal.

    Fórmula documentada pelo CDSE: 1 PU = 512 × 512 px, 3 bandas de entrada,
    1 amostra por pixel, saída de 8/16 bits. Cada fator multiplica.

    O fator que domina é o de amostras: requisição multi-temporal é cobrada por
    cena, então o custo é LINEAR no tamanho da janela temporal. Dobrar o período
    dobra a conta.

    A estimativa erra para cima de propósito — conta ``dataMask`` como banda de
    entrada. Vale gastar menos do que o previsto, não mais.
    """
    if min(output_pixels, n_bands, n_samples_per_pixel) < 0:
        raise SentinelError("parâmetros de custo não podem ser negativos")

    size_factor = max(PU_MIN_SIZE_FACTOR, output_pixels / PU_BASE_PIXELS)
    band_factor = (n_bands + EXTRA_INPUT_BANDS) / PU_BASE_BANDS
    sample_factor = max(1, n_samples_per_pixel)
    return size_factor * band_factor * sample_factor


# --------------------------------------------------------------------------- #
# Credenciais e sessão
# --------------------------------------------------------------------------- #


def credentials_from_env(env: dict[str, str] | None = None) -> tuple[str, str]:
    """Lê as credenciais do ambiente, com erro que diz o que fazer."""
    source = env if env is not None else dict(os.environ)
    client_id = (source.get(CLIENT_ID_ENV) or "").strip()
    client_secret = (source.get(CLIENT_SECRET_ENV) or "").strip()

    missing = [
        name
        for name, value in ((CLIENT_ID_ENV, client_id), (CLIENT_SECRET_ENV, client_secret))
        if not value
    ]
    if missing:
        raise SentinelError(
            f"credencial ausente: {', '.join(missing)}. Crie um OAuth client em "
            "https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings "
            "e coloque os valores no arquivo .env na raiz do repositório."
        )
    return client_id, client_secret


def build_sh_config(sentinel: SentinelConfig, env: dict[str, str] | None = None):
    """Monta o ``SHConfig`` apontado para o CDSE."""
    from sentinelhub import SHConfig

    client_id, client_secret = credentials_from_env(env)

    sh_config = SHConfig()
    sh_config.sh_client_id = client_id
    sh_config.sh_client_secret = client_secret
    sh_config.sh_base_url = sentinel.base_url
    sh_config.sh_token_url = sentinel.token_url
    return sh_config


def _data_collection(sentinel: SentinelConfig):
    """Coleção Sentinel-2 L2A redirecionada para o host do CDSE.

    Sem o ``define_from``, a biblioteca continua falando com
    ``services.sentinel-hub.com`` mesmo com ``sh_base_url`` trocado, e o
    resultado é 401 sem explicação.
    """
    from sentinelhub import DataCollection

    return DataCollection.SENTINEL2_L2A.define_from(
        "s2l2a_cdse", service_url=sentinel.base_url
    )


def aoi_in_request_crs(config: Config):
    """Extensão de aquisição reprojetada para o CRS que o Process API aceita.

    Toda a aquisição — tiles, bbox das requisições, costura — acontece neste CRS.
    A volta para ``project.crs_metric`` é feita uma vez só, no mosaico final.
    """
    from ..geo import acquisition_geometry, reproject_geometry

    aoi = acquisition_geometry(config, metric=True)
    return reproject_geometry(
        aoi, config.project.crs_metric, config.sentinel.request_crs
    )


def _sh_crs(epsg: str):
    from sentinelhub import CRS

    return CRS(epsg.split(":")[-1])


# --------------------------------------------------------------------------- #
# Catálogo
# --------------------------------------------------------------------------- #


def probe(config: Config) -> dict[str, object]:
    """Estágio ``probe-sentinel``: conta cenas e o custo do pedido, sem baixar.

    Existe porque a cota gratuita é finita (10.000 processing units/mês) e
    descobrir o tamanho do pedido depois de disparar é caro.
    """
    from sentinelhub import BBox, SentinelHubCatalog

    sentinel = config.sentinel
    aoi = aoi_in_request_crs(config)
    tiles = request_tiles(aoi.bounds, config.raster.resolution_m, sentinel.request_tile_px)

    sh_config = build_sh_config(sentinel)
    catalog = SentinelHubCatalog(config=sh_config)
    collection = _data_collection(sentinel)

    crs = _sh_crs(sentinel.request_crs)
    search = catalog.search(
        collection,
        bbox=BBox(tuple(aoi.bounds), crs=crs),
        time=(sentinel.date_start, sentinel.date_end),
        filter=f"eo:cloud_cover < {sentinel.max_cloud_cover}",
        fields={
            "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
            "exclude": [],
        },
    )

    scenes = list(search)
    dates = sorted({str(item["properties"]["datetime"])[:10] for item in scenes})
    total_px = sum(tile.pixels for tile in tiles)

    # Por pixel, o servidor vê uma amostra por passagem. Usamos as datas
    # distintas como estimativa central e o total de cenas como pior caso.
    pu_expected = estimate_processing_units(total_px, len(sentinel.bands), len(dates))
    pu_worst = estimate_processing_units(total_px, len(sentinel.bands), len(scenes))

    summary: dict[str, object] = {
        "scenes": len(scenes),
        "distinct_dates": len(dates),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "tiles": len(tiles),
        "output_pixels": total_px,
        "bands": len(sentinel.bands),
        "estimated_pu": pu_expected,
        "worst_case_pu": pu_worst,
        "budget_pu": sentinel.max_processing_units,
        "within_budget": pu_expected <= sentinel.max_processing_units,
    }

    logger.info("cenas com nuvem < %d%%: %d", sentinel.max_cloud_cover, len(scenes))
    logger.info(
        "datas distintas ...... %d (%s → %s)",
        len(dates),
        summary["first_date"],
        summary["last_date"],
    )
    logger.info("tiles de requisição .. %d", len(tiles))
    logger.info(
        "pixels de saída ...... %s por banda, %d bandas",
        f"{total_px:,}",
        len(sentinel.bands),
    )
    logger.info(
        "custo estimado ....... %.0f PU (pior caso %.0f PU), teto %.0f PU",
        pu_expected,
        pu_worst,
        sentinel.max_processing_units,
    )
    if not scenes:
        logger.warning(
            "nenhuma cena encontrada — confira a janela temporal e o filtro de nuvem"
        )
    elif pu_expected > sentinel.max_processing_units:
        logger.error(
            "ACIMA DO TETO: %.0f PU estimados contra %.0f permitidos. "
            "'acquire-sentinel' vai recusar. Encurte sentinel.date_start/date_end — "
            "o custo é linear no número de datas.",
            pu_expected,
            sentinel.max_processing_units,
        )
    logger.info(
        "o consumo de processing units cresce com pixels × bandas × cenas; "
        "a cota do plano gratuito é 10.000 PU/mês"
    )
    return summary


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def write_geotiff(
    array, bounds: Sequence[float], crs: str, resolution: float, destination: Path
) -> Path:
    """Grava um array decodificado como GeoTIFF georreferenciado.

    O ``sentinelhub`` entrega arrays sem georreferência — o recorte espacial que
    pedimos é que define a transformação. Como a bbox e o tamanho em pixels são
    nossos, a origem é exata, sem inferência.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    data = np.asarray(array)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        # tifffile devolve (altura, largura, banda); rasterio quer (banda, ...).
        data = np.transpose(data, (2, 0, 1))
    else:
        raise SentinelError(f"array com {data.ndim} dimensões, esperado 2 ou 3")

    west, _, _, north = bounds
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=from_origin(west, north, resolution, resolution),
        nodata=0,
        compress="deflate",
    ) as dst:
        dst.write(data)
    return destination


def _download_tile(
    tile: TileRequest,
    config: Config,
    sh_config,
    collection,
    evalscript: str,
    destination: Path,
) -> Path:
    from sentinelhub import BBox, MimeType, SentinelHubRequest

    sentinel = config.sentinel
    crs = _sh_crs(sentinel.request_crs)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=collection,
                time_interval=(sentinel.date_start, sentinel.date_end),
                maxcc=sentinel.max_cloud_cover / 100.0,
                other_args={"processing": {"upsampling": "BILINEAR"}},
            )
        ],
        responses=[
            SentinelHubRequest.output_response(RESPONSE_BANDS, MimeType.TIFF),
            SentinelHubRequest.output_response(RESPONSE_MASK, MimeType.TIFF),
        ],
        bbox=BBox(tile.bounds, crs=crs),
        size=(tile.width_px, tile.height_px),
        config=sh_config,
        data_folder=str(destination),
    )

    logger.info(
        "%s: %d × %d px, bbox %s", tile.name, tile.width_px, tile.height_px, tile.bounds
    )

    # save_data=True guarda a resposta crua em <destino>/<hash>/response.tar e,
    # numa próxima execução, o hash do payload faz o cliente ler do disco em vez
    # de baixar. Enquanto o payload não mudar, repetir o estágio não custa PU.
    results = request.get_data(save_data=True, show_progress=False)
    if not results:
        raise SentinelError(f"{tile.name}: resposta vazia do Process API")

    # Duas saídas no pedido ⇒ a resposta vem como tar e é decodificada num
    # dicionário {nome do arquivo: array}.
    payload = results[0]
    if not isinstance(payload, dict):
        raise SentinelError(
            f"{tile.name}: resposta inesperada ({type(payload).__name__}); "
            "esperado um dicionário de arquivos"
        )

    for identifier in (RESPONSE_BANDS, RESPONSE_MASK):
        key = f"{identifier}.tif"
        if key not in payload:
            raise SentinelError(
                f"{tile.name}: '{key}' ausente na resposta "
                f"(veio: {sorted(payload)})"
            )
        write_geotiff(
            payload[key],
            tile.bounds,
            sentinel.request_crs,
            config.raster.resolution_m,
            destination / key,
        )

    return destination


def acquire(config: Config) -> Path:
    """Estágio ``acquire-sentinel``: baixa os tiles e costura o mosaico."""
    sentinel = config.sentinel
    aoi = aoi_in_request_crs(config)
    tiles = request_tiles(aoi.bounds, config.raster.resolution_m, sentinel.request_tile_px)
    if not tiles:
        raise SentinelError("AOI não gerou nenhum tile de requisição")

    # Consulta o catálogo antes de baixar: o custo é linear no número de cenas e
    # estourar a cota trava o projeto até o mês virar.
    summary = probe(config)
    estimated = float(summary["estimated_pu"])  # type: ignore[arg-type]
    if not summary["within_budget"]:
        raise SentinelError(
            f"pedido estimado em {estimated:.0f} PU, acima do teto de "
            f"{sentinel.max_processing_units:.0f} definido em "
            "'sentinel.max_processing_units'. O custo é linear no número de datas: "
            "encurte a janela em 'sentinel.date_start'/'date_end', ou aperte "
            "'sentinel.max_cloud_cover'. Se o gasto for intencional, suba o teto "
            "no YAML — conscientemente."
        )
    if summary["scenes"] == 0:
        raise SentinelError(
            "nenhuma cena no catálogo para esta janela e filtro de nuvem — "
            "não há o que compor"
        )

    evalscript = build_evalscript(sentinel.bands, sentinel.invalid_scl_classes)
    sh_config = build_sh_config(sentinel)
    collection = _data_collection(sentinel)

    raw_dir = config.path("data_raw") / "sentinel2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "evalscript.js").write_text(evalscript, encoding="utf-8")

    logger.info("baixando %d tile(s) para %s", len(tiles), config.display_path(raw_dir))
    for tile in tiles:
        _download_tile(tile, config, sh_config, collection, evalscript, raw_dir / tile.name)

    mosaic = config.path("data_interim") / "s2_median.tif"
    written = merge_tiles(raw_dir, mosaic, aoi_bounds=aoi.bounds, config=config)
    logger.info("mosaico escrito em %s", config.display_path(written))
    return written


# --------------------------------------------------------------------------- #
# Costura
# --------------------------------------------------------------------------- #


def merge_tiles(
    raw_dir: Path,
    destination: Path,
    aoi_bounds: Sequence[float],
    config: Config,
) -> Path:
    """Costura os tiles e reprojeta o mosaico para o CRS de análise.

    ``aoi_bounds`` vem no CRS da requisição (``sentinel.request_crs``), que é
    onde os tiles foram baixados. A saída sai em ``project.crs_metric``.
    """
    import numpy as np
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject, transform_bounds

    sources = sorted(raw_dir.rglob(f"{RESPONSE_BANDS}.tif"))
    if not sources:
        raise SentinelError(
            f"nenhum '{RESPONSE_BANDS}.tif' encontrado em {raw_dir} — o download falhou?"
        )

    resolution = config.raster.resolution_m
    request_crs = config.sentinel.request_crs
    target_crs = config.project.crs_metric

    logger.info("costurando %d arquivo(s) em %s", len(sources), request_crs)
    handles = [rasterio.open(path) for path in sources]
    try:
        src_array, src_transform = merge(
            handles,
            bounds=snap_bounds(aoi_bounds, resolution),
            res=resolution,
            # 'first' basta: os tiles não se sobrepõem, então não há conflito real.
            method="first",
            nodata=0,
        )
        dtype = handles[0].profile["dtype"]
    finally:
        for handle in handles:
            handle.close()

    # Grade de saída no CRS de análise, alinhada à resolução. Derivada dos
    # próprios limites recebidos, para que a função não dependa de estado externo.
    if request_crs == target_crs:
        target_bounds = snap_bounds(aoi_bounds, resolution)
    else:
        target_bounds = snap_bounds(
            transform_bounds(request_crs, target_crs, *aoi_bounds), resolution
        )
    west, south, east, north = target_bounds
    width = round((east - west) / resolution)
    height = round((north - south) / resolution)
    transform = from_origin(west, north, resolution, resolution)

    if request_crs == target_crs:
        array = src_array
    else:
        logger.info("reprojetando %s → %s", request_crs, target_crs)
        array = np.zeros((src_array.shape[0], height, width), dtype=dtype)
        reproject(
            source=src_array,
            destination=array,
            src_transform=src_transform,
            src_crs=request_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            # Vizinho mais próximo de propósito: SIRGAS 2000 e WGS84 diferem por
            # poucos centímetros no Brasil, muito abaixo do pixel de 10 m. Não há
            # deslocamento real a corrigir, e interpolar só suavizaria as bordas
            # que a U-Net precisa enxergar nítidas.
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )

    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": dtype,
        "transform": transform,
        "crs": target_crs,
        "nodata": 0,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(array)
        dst.update_tags(
            reflectance_scale=str(REFLECTANCE_SCALE),
            composite=config.sentinel.composite,
            date_start=config.sentinel.date_start,
            date_end=config.sentinel.date_end,
            max_cloud_cover=str(config.sentinel.max_cloud_cover),
            source="Copernicus Sentinel-2 L2A via CDSE",
            request_crs=request_crs,
            analysis_crs=target_crs,
        )
        for position, band in enumerate(config.sentinel.bands, start=1):
            dst.set_band_description(position, band)

    return destination

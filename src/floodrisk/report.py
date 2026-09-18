"""Estágio ``report``: prepara os dados que o site publicado consome.

O site é estático no GitHub Pages: não há servidor para consultar, então tudo
que a página mostra precisa estar materializado em arquivo antes do build. Este
estágio é a ponte entre o pipeline e o site.

**O problema que ele resolve.** A grade de suscetibilidade sai do
``susceptibility`` com 5,6 MB de GeoJSON. Isso não é aceitável numa página que
alguém vai abrir do celular ou do wifi de uma sala de defesa: são vários segundos
de tela em branco antes do mapa aparecer, e é o tipo de falha que acontece
exatamente na hora da apresentação. Duas reduções resolvem sem perder nada que o
mapa use:

1. **Precisão das coordenadas.** O geopandas escreve ~15 casas decimais, o que em
   latitude são frações de nanômetro. Cinco casas dão ~1 m — para células de
   200 m, é precisão de sobra, e corta a maior parte do arquivo.
2. **Propriedades.** O GeoPackage guarda tudo para análise; o site usa quatro
   campos. Os demais viajariam 11.239 vezes sem ninguém ler.

**Números nunca são escritos à mão no site.** As métricas vão para um
``metrics.json`` que as páginas leem. Número copiado à mão em texto é número que
fica desatualizado no primeiro retreino, e ninguém percebe até a banca conferir
contra o relatório.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import artifacts
from .config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "WEB_PROPERTIES",
    "ReportError",
    "build",
    "compact_geojson",
    "round_coordinates",
]

#: Propriedades que o mapa usa, e o nome curto com que viajam. O resto fica no
#: GeoPackage, para análise.
#:
#: Os nomes curtos não são economia de digitação: ``"impervious_mean":0.9551``
#: gasta 24 caracteres e se repete 11.239 vezes, uma vez por célula. Só os nomes
#: das quatro chaves respondem por centenas de kB do arquivo. O mapeamento fica
#: aqui, explícito, e o JavaScript do site o consome pelo ``metrics.json`` em vez
#: de ter as letras espalhadas pelo código.
WEB_PROPERTIES = {
    "class_index": "c",
    "susceptibility": "s",
    "impervious_mean": "i",
    "slope_mean_deg": "d",
}

#: Casas decimais em grau. 5 casas ≈ 1,1 m — precisão de sobra para células de
#: 200 m, e é onde mora a maior parte da economia de tamanho.
COORDINATE_DECIMALS = 5

#: Precisão de cada propriedade, pelo que o popup mostra. Guardar quatro casas
#: de uma declividade exibida como "2,4°" é pagar transporte por dígito que
#: ninguém vê.
PROPERTY_DECIMALS = {
    "susceptibility": 3,
    "impervious_mean": 3,
    "slope_mean_deg": 2,
}

#: Acima disto o estágio avisa. O limite considera que o GitHub Pages serve com
#: gzip (GeoJSON comprime ~6x) e que o mapa renderiza em canvas — o gargalo real
#: não é a rede, é a contagem de polígonos.
SIZE_WARNING_BYTES = 4_000_000


class ReportError(RuntimeError):
    """Insumo ausente na preparação dos dados do site."""


def round_coordinates(value, decimals: int = COORDINATE_DECIMALS):
    """Arredonda recursivamente toda coordenada de uma geometria GeoJSON.

    A estrutura de coordenadas é aninhada a profundidades diferentes conforme o
    tipo (Point, Polygon, MultiPolygon), então a recursão desce até achar
    números em vez de tratar cada tipo separadamente.
    """
    if isinstance(value, list):
        return [round_coordinates(item, decimals) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), decimals)
    return value


def _round_property(name, value):
    """Arredonda pela precisão declarada para aquela propriedade.

    Inteiro passa intacto: ``class_index`` indexa um array de cores no
    JavaScript, e virar float quebraria o acesso.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(value, int):
        return value
    return round(float(value), PROPERTY_DECIMALS.get(name, 4))


def compact_geojson(data, keep=WEB_PROPERTIES, decimals: int = COORDINATE_DECIMALS):
    """Devolve o GeoJSON com coordenadas arredondadas e propriedades enxutas.

    ``keep`` mapeia o nome original ao nome curto com que a propriedade viaja.
    """
    if data.get("type") != "FeatureCollection":
        raise ReportError(
            f"esperado FeatureCollection, veio {data.get('type')!r}"
        )
    features = data.get("features")
    if not isinstance(features, list):
        raise ReportError("o GeoJSON não traz uma lista de feições")

    keep = dict(keep)
    compacted = []
    for feature in features:
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        compacted.append(
            {
                "type": "Feature",
                "properties": {
                    short: _round_property(name, properties[name])
                    for name, short in keep.items()
                    if name in properties
                },
                "geometry": {
                    "type": geometry["type"],
                    "coordinates": round_coordinates(geometry["coordinates"], decimals),
                },
            }
        )

    if not compacted:
        raise ReportError("nenhuma feição sobrou depois da compactação")

    out = {"type": "FeatureCollection", "features": compacted}
    if "crs" in data:
        out["crs"] = data["crs"]
    return out


def _read_json(path: Path, what: str):
    if not path.exists():
        raise ReportError(f"{what} ausente: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload, compact: bool = True) -> int:
    """Grava e devolve o tamanho em bytes.

    Sem indentação nem espaço depois dos separadores: num arquivo com 11.239
    feições, o espaçamento sozinho é centenas de kB que o navegador baixa e
    descarta.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    separators = (",", ":") if compact else None
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=separators,
        indent=None if compact else 2,
    )
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _collect_metrics(config: Config) -> dict:
    """Reúne os números que o texto do site cita, para nenhum ser digitado."""
    metrics: dict = {"project": config.project.name}

    evaluation = artifacts.evaluation_report(config)
    if evaluation.exists():
        report = _read_json(evaluation, "relatório de avaliação")
        tuned = report.get("test", {}).get("tuned", {})
        metrics["model"] = {
            "encoder": report.get("encoder"),
            "patch_size": report.get("patch_size"),
            "threshold": report.get("tuned_threshold"),
            "test_dice": round(tuned.get("dice", 0.0), 4),
            "test_iou": round(tuned.get("iou", 0.0), 4),
            "test_precision": round(tuned.get("precision", 0.0), 4),
            "test_recall": round(tuned.get("recall", 0.0), 4),
            "validation_dice": round(
                report.get("validation", {}).get("tuned", {}).get("dice", 0.0), 4
            ),
        }
        by_aoi = report.get("test_by_aoi_fraction")
        if by_aoi:
            metrics["model"]["by_coverage"] = [
                {
                    "low": row["low"],
                    "high": row["high"],
                    "patches": row["patches"],
                    "dice": round(row.get("dice", 0.0), 4),
                    "positive_rate": round(row.get("positive_rate", 0.0), 4),
                }
                for row in by_aoi
                if row["patches"]
            ]

    split = artifacts.split_record(config)
    if split.exists():
        record = _read_json(split, "registro do split")
        metrics["split"] = {
            "strategy": record.get("strategy"),
            "block_size_m": record.get("block_size_m"),
            "patches": record.get("kept"),
            "chosen_seed": record.get("chosen_seed"),
            "prevalence_spread_pp": record.get("prevalence_spread_pp"),
        }

    metrics["scenarios"] = {
        "source": config.risk_scenarios.source,
        "tiers": [
            {
                "name": tier.name,
                "label": tier.label,
                "hourly_mm": tier.hourly_mm,
                "daily_mm": tier.daily_mm,
                "inmet_alert": tier.inmet_alert,
            }
            for tier in config.risk_scenarios.tiers
        ],
    }
    metrics["grid"] = {
        "cell_size_m": config.grid.cell_size_m,
        "classes": list(_class_labels()),
        # O mapa lê as propriedades por estas chaves. Fica aqui para que os
        # nomes curtos do GeoJSON tenham uma definição única e legível.
        "properties": dict(WEB_PROPERTIES),
    }
    metrics["attribution"] = {
        "forecast": config.forecast.attribution,
        "sentinel": "Copernicus Sentinel-2 L2A (ESA / CDSE)",
        "worldcover": "ESA WorldCover 2021 v200",
        "dem": "Copernicus DEM GLO-30",
        "streets": "GeoCuritiba / IPPUC — Prefeitura de Curitiba",
        "boundary": "Malhas municipais IBGE",
    }
    return metrics


def _class_labels():
    from .features.susceptibility import CLASS_LABELS

    return CLASS_LABELS


def build(config: Config) -> Path:
    """Estágio ``report``: materializa os dados do site em ``site/data``."""
    destination = config.root / "site" / "data"
    destination.mkdir(parents=True, exist_ok=True)

    source = artifacts.susceptibility_geojson(config)
    if not source.exists():
        raise ReportError(
            f"camada ausente: {config.display_path(source)}. "
            "Rode 'susceptibility' antes."
        )
    original = source.stat().st_size
    compacted = compact_geojson(_read_json(source, "camada de suscetibilidade"))
    written = _write_json(destination / "susceptibility.geojson", compacted)
    logger.info(
        "suscetibilidade: %.1f MB → %.1f MB (%.0f%% menor), %d células",
        original / 1e6,
        written / 1e6,
        100 * (1 - written / original),
        len(compacted["features"]),
    )
    if written > SIZE_WARNING_BYTES:
        logger.warning(
            "a camada tem %.1f MB mesmo compactada. O GitHub Pages serve com "
            "gzip (GeoJSON comprime ~6x), então a rede aguenta; o gargalo passa "
            "a ser desenhar %d polígonos. Se o mapa travar, aumente "
            "'grid.cell_size_m' para 300 m.",
            written / 1e6,
            len(compacted["features"]),
        )

    forecast = config.path("data_raw") / "forecast" / "forecast_snapshot.json"
    if forecast.exists():
        payload = _read_json(forecast, "snapshot da previsão")
        size = _write_json(destination / "forecast.json", payload, compact=False)
        logger.info("previsão: %.0f kB (fallback do site)", size / 1000)
    else:
        logger.warning(
            "snapshot da previsão ausente; o site vai depender da chamada ao vivo "
            "da Open-Meteo. Rode 'acquire-forecast'."
        )

    try:
        from .basemap import build_basemap

        metrics_basemap = build_basemap(config, destination / "sentinel_rgb.jpg")
    except Exception as exc:  # pragma: no cover - depende do mosaico em disco
        # Falta de camada de satélite não pode derrubar o estágio: o site cai
        # para o mapa de ruas e continua utilizável.
        logger.warning("camada de satélite indisponível: %s", exc)
        metrics_basemap = None

    metrics = _collect_metrics(config)
    if metrics_basemap:
        metrics["basemap"] = metrics_basemap
    _write_json(destination / "metrics.json", metrics, compact=False)
    logger.info("métricas escritas em %s", config.display_path(destination / "metrics.json"))

    if "model" not in metrics:
        logger.warning(
            "sem 'reports/evaluation.json': o site ficará sem os números do "
            "modelo. Rode 'evaluate'."
        )

    logger.info("dados do site prontos em %s", config.display_path(destination))
    return destination

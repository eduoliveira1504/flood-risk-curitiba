"""Cliente Open-Meteo — previsão operacional e arquivo histórico de chuva.

Papel da chuva no projeto: ela NÃO espacializa risco. A célula do modelo de
reanálise tem 9–11 km e Curitiba tem ~21 × 34 km, então a chuva entra como um
valor único da cidade, cujo papel é apenas **selecionar** qual camada de risco
pré-materializada o site exibe. A variação espacial do mapa vem do território.

Dois endpoints, dois usos:

- ``forecast`` → previsão operacional, alimenta a visão de previsão do site e o
  ``forecast_snapshot.json`` congelado no build (fallback para quando a chamada
  do navegador falhar — API fora do ar, sem rede na sala da defesa, ou ausência
  de CORS).
- ``archive`` → reanálise histórica, usada na calibração dos patamares e na
  validação contra as ocorrências registradas.

Licença: os dados são CC-BY 4.0. ``forecast.attribution`` na configuração carrega
o crédito obrigatório e é propagado para o snapshot, para que o site não possa ser
publicado sem ele.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from ..config import (
    INMET_DAILY_WINDOW_H,
    INMET_HOURLY_WINDOW_H,
    Config,
    ForecastConfig,
    RiskScenariosConfig,
    ScenarioTier,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ForecastError",
    "OpenMeteoClient",
    "RainSeries",
    "build_snapshot",
    "write_snapshot",
]

_USER_AGENT = "flood-risk-curitiba/0.1 (TCC, FAE Centro Universitario, Curitiba BR; academic use)"

# Códigos que valem uma nova tentativa: 429 é limite de taxa, 5xx é falha do lado deles.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class ForecastError(RuntimeError):
    """Falha ao obter ou interpretar a resposta da Open-Meteo."""


@dataclass(frozen=True)
class RainSeries:
    """Série horária de precipitação, em mm por hora."""

    times: list[str]
    precipitation_mm: list[float]
    latitude: float
    longitude: float
    timezone: str
    source: str

    def __post_init__(self) -> None:
        if len(self.times) != len(self.precipitation_mm):
            raise ForecastError(
                f"série inconsistente: {len(self.times)} instantes para "
                f"{len(self.precipitation_mm)} valores de precipitação"
            )

    def __len__(self) -> int:
        return len(self.times)

    @property
    def total_mm(self) -> float:
        return sum(self.precipitation_mm)

    def max_accumulation(self, window_h: int) -> float:
        """Maior acumulado em qualquer janela deslizante de ``window_h`` horas.

        É o máximo, não o total: o que dispara alagamento é intensidade num
        intervalo curto, não volume espalhado por dias. 40 mm em 6 h alaga;
        os mesmos 40 mm em duas semanas não.
        """
        if window_h < 1:
            raise ValueError("window_h precisa ser >= 1")
        values = self.precipitation_mm
        if not values:
            return 0.0
        if window_h >= len(values):
            return float(sum(values))

        current = float(sum(values[:window_h]))
        best = current
        for index in range(window_h, len(values)):
            current += values[index] - values[index - window_h]
            best = max(best, current)
        return best

    def accumulation_series(self, window_h: int) -> list[float]:
        """Acumulado móvel alinhado ao FIM de cada janela.

        As primeiras ``window_h - 1`` posições ficam com o acumulado parcial
        disponível até ali, para que a série tenha o mesmo comprimento de
        ``times`` e possa ser plotada direto contra ele.
        """
        if window_h < 1:
            raise ValueError("window_h precisa ser >= 1")
        out: list[float] = []
        running = 0.0
        for index, value in enumerate(self.precipitation_mm):
            running += value
            if index >= window_h:
                running -= self.precipitation_mm[index - window_h]
            out.append(running)
        return out


class OpenMeteoClient:
    """Acesso às APIs de previsão e de arquivo da Open-Meteo.

    Sem API key: o plano gratuito não exige uma para uso não-comercial.
    """

    def __init__(
        self,
        config: ForecastConfig,
        session: requests.Session | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self._config = config
        self._session = session or requests.Session()
        self._session.headers.setdefault("User-Agent", _USER_AGENT)
        self._sleep = sleep

    # -- público ----------------------------------------------------------- #

    def forecast(self, horizon_hours: int | None = None) -> RainSeries:
        """Previsão horária de precipitação para o ponto de referência."""
        hours = horizon_hours or self._config.horizon_hours
        # A API trabalha em dias; pedimos o teto e recortamos depois.
        days = min(16, -(-hours // 24))
        payload = self._get(
            self._config.forecast_endpoint,
            {
                "latitude": self._config.reference_point.lat,
                "longitude": self._config.reference_point.lon,
                "hourly": "precipitation",
                "forecast_days": days,
                "timezone": self._config.timezone,
            },
        )
        series = self._parse(payload, source="open-meteo:forecast")
        if len(series) <= hours:
            return series
        return RainSeries(
            times=series.times[:hours],
            precipitation_mm=series.precipitation_mm[:hours],
            latitude=series.latitude,
            longitude=series.longitude,
            timezone=series.timezone,
            source=series.source,
        )

    def archive(self, start_date: str, end_date: str) -> RainSeries:
        """Reanálise histórica de precipitação horária. Datas em ``YYYY-MM-DD``."""
        payload = self._get(
            self._config.archive_endpoint,
            {
                "latitude": self._config.reference_point.lat,
                "longitude": self._config.reference_point.lon,
                "hourly": "precipitation",
                "start_date": start_date,
                "end_date": end_date,
                "timezone": self._config.timezone,
            },
        )
        return self._parse(payload, source="open-meteo:archive")

    # -- interno ----------------------------------------------------------- #

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                response = self._session.get(
                    url, params=params, timeout=self._config.request_timeout_s
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("tentativa %d falhou na rede: %s", attempt, exc)
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise ForecastError(f"resposta não é JSON válido: {exc}") from exc

                # A Open-Meteo devolve {"error": true, "reason": "..."} em 400.
                if response.status_code not in _RETRYABLE_STATUS:
                    raise ForecastError(
                        f"HTTP {response.status_code} em {url}: {_reason(response)}"
                    )

                last_error = ForecastError(
                    f"HTTP {response.status_code}: {_reason(response)}"
                )
                logger.warning(
                    "tentativa %d recebeu HTTP %d (recuperável)", attempt, response.status_code
                )

            if attempt < self._config.max_retries:
                backoff = 2.0 ** (attempt - 1)
                logger.info("aguardando %.0fs antes de tentar de novo", backoff)
                self._sleep(backoff)

        raise ForecastError(
            f"falhou após {self._config.max_retries} tentativas em {url}"
        ) from last_error

    def _parse(self, payload: Any, source: str) -> RainSeries:
        if not isinstance(payload, dict):
            raise ForecastError(f"payload inesperado: {type(payload).__name__}")
        if payload.get("error"):
            raise ForecastError(f"API recusou a consulta: {payload.get('reason')}")

        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise ForecastError("resposta sem o bloco 'hourly'")

        times = hourly.get("time")
        precipitation = hourly.get("precipitation")
        if not isinstance(times, list) or not isinstance(precipitation, list):
            raise ForecastError("'hourly.time' ou 'hourly.precipitation' ausente")

        # Hora sem dado vem como null; tratar como zero seria inventar ausência de
        # chuva, mas propagar None quebra a soma. Zeramos e registramos.
        holes = sum(1 for value in precipitation if value is None)
        if holes:
            logger.warning("%d hora(s) sem precipitação na resposta, tratadas como 0.0", holes)

        return RainSeries(
            times=[str(t) for t in times],
            precipitation_mm=[float(v) if v is not None else 0.0 for v in precipitation],
            latitude=float(payload.get("latitude", self._config.reference_point.lat)),
            longitude=float(payload.get("longitude", self._config.reference_point.lon)),
            timezone=str(payload.get("timezone", self._config.timezone)),
            source=source,
        )


def _reason(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and "reason" in body:
        return str(body["reason"])
    return str(body)[:200]


# --------------------------------------------------------------------------- #
# Snapshot consumido pelo site
# --------------------------------------------------------------------------- #


def build_snapshot(
    series: RainSeries,
    scenarios: RiskScenariosConfig,
    forecast_config: ForecastConfig,
) -> dict[str, Any]:
    """Monta o dicionário que o site lê como fallback da previsão.

    Inclui o cenário já resolvido para que a página funcione mesmo sem executar
    a classificação no navegador, e carrega a atribuição CC-BY obrigatória.
    """
    window = forecast_config.accumulation_window_h
    peak = series.max_accumulation(window)

    # Os dois critérios do INMET, avaliados nas janelas que os definem.
    hourly_peak = series.max_accumulation(INMET_HOURLY_WINDOW_H)
    daily_peak = series.max_accumulation(INMET_DAILY_WINDOW_H)
    tier: ScenarioTier = scenarios.classify(hourly_peak, daily_peak)

    # Qual critério puxou o cenário. Vai para o site porque "chuva forte" sem
    # dizer se é intensidade ou volume não ajuda ninguém a decidir nada.
    if tier.hourly_mm > 0 and hourly_peak >= tier.hourly_mm:
        triggered_by = "intensidade horária"
    elif tier.daily_mm > 0:
        triggered_by = "acumulado em 24 h"
    else:
        triggered_by = "nenhum critério atingido"

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": series.source,
        "attribution": forecast_config.attribution,
        "licence": "CC-BY 4.0",
        "point": {"lat": series.latitude, "lon": series.longitude},
        "timezone": series.timezone,
        "accumulation_window_h": window,
        "peak_accumulation_mm": round(peak, 2),
        "peak_hourly_mm": round(hourly_peak, 2),
        "peak_daily_mm": round(daily_peak, 2),
        "total_mm": round(series.total_mm, 2),
        "scenario": {
            "name": tier.name,
            "label": tier.label,
            "hourly_mm": tier.hourly_mm,
            "daily_mm": tier.daily_mm,
            "inmet_alert": tier.inmet_alert,
            "triggered_by": triggered_by,
        },
        "scenario_tiers": [
            {
                "name": t.name,
                "label": t.label,
                "hourly_mm": t.hourly_mm,
                "daily_mm": t.daily_mm,
                "inmet_alert": t.inmet_alert,
            }
            for t in scenarios.tiers
        ],
        "scenario_source": scenarios.source,
        "justification_pending": scenarios.justification_pending,
        "hourly": {
            "time": series.times,
            "precipitation_mm": [round(v, 2) for v in series.precipitation_mm],
            "accumulation_mm": [
                round(v, 2) for v in series.accumulation_series(window)
            ],
        },
    }


def write_snapshot(snapshot: dict[str, Any], destination: Path) -> Path:
    """Grava o snapshot como JSON legível. Devolve o caminho escrito."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def acquire(config: Config) -> Path:
    """Estágio ``acquire-forecast``: busca a previsão e materializa o snapshot."""
    client = OpenMeteoClient(config.forecast)
    series = client.forecast()
    logger.info(
        "previsão obtida: %d horas, total %.1f mm, pico %.1f mm em %dh",
        len(series),
        series.total_mm,
        series.max_accumulation(config.forecast.accumulation_window_h),
        config.forecast.accumulation_window_h,
    )

    snapshot = build_snapshot(series, config.risk_scenarios, config.forecast)
    logger.info(
        "pico horário %.1f mm/h · pico em 24 h %.1f mm",
        snapshot["peak_hourly_mm"],
        snapshot["peak_daily_mm"],
    )
    logger.info(
        "cenário selecionado: %s (%s, aviso INMET %s)",
        snapshot["scenario"]["label"],
        snapshot["scenario"]["triggered_by"],
        snapshot["scenario"]["inmet_alert"],
    )
    if snapshot["justification_pending"]:
        logger.warning(
            "patamares de chuva ainda sem procedência declarada "
            "(risk_scenarios.justification_pending = true)"
        )

    destination = config.path("data_raw") / "forecast" / "forecast_snapshot.json"
    written = write_snapshot(snapshot, destination)
    logger.info("snapshot gravado em %s", config.display_path(written))
    return written

from __future__ import annotations

import json

import pytest
import requests

from floodrisk.acquisition.forecast import (
    ForecastError,
    OpenMeteoClient,
    RainSeries,
    build_snapshot,
    write_snapshot,
)
from floodrisk.config import load_config

# --------------------------------------------------------------------------- #
# Dublês — nenhum teste toca a rede
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if self._payload is _INVALID_JSON:
            raise ValueError("no json")
        return self._payload


_INVALID_JSON = object()


class FakeSession:
    """Devolve respostas roteirizadas e registra as chamadas feitas."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.headers: dict[str, str] = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def payload(times: list[str], precipitation: list[float | None]) -> dict:
    return {
        "latitude": -25.43,
        "longitude": -49.27,
        "timezone": "America/Sao_Paulo",
        "hourly": {"time": times, "precipitation": precipitation},
    }


def hours(n: int) -> list[str]:
    return [f"2026-09-14T{i:02d}:00" for i in range(n)]


@pytest.fixture
def config():
    return load_config()


def client_with(config, responses):
    return OpenMeteoClient(config.forecast, session=FakeSession(responses), sleep=lambda _: None)


# --------------------------------------------------------------------------- #
# RainSeries
# --------------------------------------------------------------------------- #


def series_of(values: list[float]) -> RainSeries:
    return RainSeries(
        times=hours(len(values)),
        precipitation_mm=values,
        latitude=-25.43,
        longitude=-49.27,
        timezone="America/Sao_Paulo",
        source="test",
    )


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ForecastError, match="inconsistente"):
        RainSeries(
            times=hours(3),
            precipitation_mm=[1.0, 2.0],
            latitude=0.0,
            longitude=0.0,
            timezone="UTC",
            source="test",
        )


def test_max_accumulation_finds_the_worst_window():
    values = [0.0, 1.0, 1.0, 2.0, 3.0, 10.0, 1.0, 0.0, 0.0, 0.0]
    # Janelas de 6h: [0..5]=17, [1..6]=18, [2..7]=17, [3..8]=16, [4..9]=14.
    # A pior não começa no primeiro instante — é isso que o teste protege.
    assert series_of(values).max_accumulation(6) == pytest.approx(18.0)


def test_max_accumulation_is_peak_not_total():
    """40 mm em 6h e 40 mm espalhados por 40h têm o mesmo total e riscos diferentes."""
    burst = series_of([0.0] * 10 + [40.0] + [0.0] * 10)
    drizzle = series_of([2.0] * 20)
    assert burst.total_mm == pytest.approx(drizzle.total_mm)
    assert burst.max_accumulation(6) > drizzle.max_accumulation(6)


def test_max_accumulation_window_larger_than_series():
    assert series_of([1.0, 2.0]).max_accumulation(24) == pytest.approx(3.0)


def test_max_accumulation_empty_series():
    assert series_of([]).max_accumulation(6) == 0.0


def test_max_accumulation_rejects_zero_window():
    with pytest.raises(ValueError):
        series_of([1.0]).max_accumulation(0)


def test_accumulation_series_matches_length_and_peak():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    series = series_of(values)
    rolling = series.accumulation_series(3)
    assert len(rolling) == len(values)
    assert max(rolling) == pytest.approx(series.max_accumulation(3))
    assert rolling[0] == pytest.approx(1.0)
    assert rolling[2] == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# Cliente HTTP
# --------------------------------------------------------------------------- #


def test_forecast_parses_and_truncates_to_horizon(config):
    client = client_with(config, [FakeResponse(200, payload(hours(72), [0.5] * 72))])
    series = client.forecast(horizon_hours=48)
    assert len(series) == 48
    assert series.source == "open-meteo:forecast"


def test_forecast_sends_no_api_key(config):
    session = FakeSession([FakeResponse(200, payload(hours(24), [0.0] * 24))])
    OpenMeteoClient(config.forecast, session=session, sleep=lambda _: None).forecast(24)
    sent = session.calls[0]["params"]
    assert "apikey" not in sent and "key" not in sent
    assert sent["hourly"] == "precipitation"
    assert sent["timezone"] == "America/Sao_Paulo"


def test_nulls_become_zero_without_breaking_the_sum(config):
    client = client_with(config, [FakeResponse(200, payload(hours(4), [1.0, None, 2.0, None]))])
    series = client.forecast(horizon_hours=4)
    assert series.precipitation_mm == [1.0, 0.0, 2.0, 0.0]
    assert series.total_mm == pytest.approx(3.0)


def test_retries_on_429_then_succeeds(config):
    client = client_with(
        config,
        [
            FakeResponse(429, {"reason": "limite"}),
            FakeResponse(200, payload(hours(24), [0.0] * 24)),
        ],
    )
    assert len(client.forecast(horizon_hours=24)) == 24


def test_retries_on_network_error_then_succeeds(config):
    client = client_with(
        config,
        [
            requests.ConnectionError("boom"),
            FakeResponse(200, payload(hours(24), [0.0] * 24)),
        ],
    )
    assert len(client.forecast(horizon_hours=24)) == 24


def test_gives_up_after_max_retries(config):
    responses = [FakeResponse(503, {"reason": "indisponível"})] * config.forecast.max_retries
    client = client_with(config, responses)
    with pytest.raises(ForecastError, match="após"):
        client.forecast(horizon_hours=24)


def test_400_is_not_retried(config):
    session = FakeSession(
        [FakeResponse(400, {"error": True, "reason": "parâmetro inválido"})]
    )
    client = OpenMeteoClient(config.forecast, session=session, sleep=lambda _: None)
    with pytest.raises(ForecastError, match="parâmetro inválido"):
        client.forecast(horizon_hours=24)
    assert len(session.calls) == 1


def test_api_level_error_flag_is_surfaced(config):
    client = client_with(config, [FakeResponse(200, {"error": True, "reason": "fora de faixa"})])
    with pytest.raises(ForecastError, match="fora de faixa"):
        client.forecast(horizon_hours=24)


def test_missing_hourly_block_is_rejected(config):
    client = client_with(config, [FakeResponse(200, {"latitude": 0, "longitude": 0})])
    with pytest.raises(ForecastError, match="hourly"):
        client.forecast(horizon_hours=24)


def test_non_json_body_is_rejected(config):
    client = client_with(config, [FakeResponse(200, _INVALID_JSON, text="<html>")])
    with pytest.raises(ForecastError, match="JSON"):
        client.forecast(horizon_hours=24)


def test_archive_uses_the_archive_endpoint(config):
    session = FakeSession([FakeResponse(200, payload(hours(48), [0.0] * 48))])
    client = OpenMeteoClient(config.forecast, session=session, sleep=lambda _: None)
    series = client.archive("2024-01-01", "2024-01-02")
    assert session.calls[0]["url"] == config.forecast.archive_endpoint
    assert session.calls[0]["params"]["start_date"] == "2024-01-01"
    assert series.source == "open-meteo:archive"


# --------------------------------------------------------------------------- #
# Cenários e snapshot
# --------------------------------------------------------------------------- #


def test_classify_picks_the_highest_tier_reached(config):
    scenarios = config.risk_scenarios
    assert scenarios.classify(0.0, 0.0).name == "leve"
    assert scenarios.classify(19.9, 49.9).name == "leve"
    assert scenarios.classify(20.0, 0.0).name == "forte"
    assert scenarios.classify(0.0, 50.0).name == "forte"
    assert scenarios.classify(60.0, 0.0).name == "critica"
    assert scenarios.classify(0.0, 100.0).name == "critica"


def test_either_criterion_alone_can_raise_the_scenario(config):
    """É o "ou" do INMET: 30 mm numa hora é crítico mesmo sem volume no dia."""
    scenarios = config.risk_scenarios
    assert scenarios.classify(70.0, 10.0).name == "critica"
    assert scenarios.classify(2.0, 120.0).name == "critica"


def test_volume_spread_over_a_day_is_not_the_same_as_intensity(config):
    """120 mm espalhados em 24 h sem pico horário não é o mesmo que 120 mm em 2 h.

    É exatamente a distinção que um limiar único não conseguia fazer, e a razão
    de o INMET publicar dois critérios.
    """
    scenarios = config.risk_scenarios
    gentle = scenarios.classify(5.0, 40.0)
    intense = scenarios.classify(65.0, 40.0)
    assert gentle.name == "leve"
    assert intense.name == "critica"


def test_classify_rejects_negative(config):
    with pytest.raises(ValueError):
        config.risk_scenarios.classify(-1.0, 0.0)
    with pytest.raises(ValueError):
        config.risk_scenarios.classify(0.0, -1.0)


def test_snapshot_carries_attribution_and_resolved_scenario(config):
    series = series_of([0.0] * 10 + [25.0] + [0.0] * 10)
    snapshot = build_snapshot(series, config.risk_scenarios, config.forecast)

    assert snapshot["attribution"] == config.forecast.attribution
    assert snapshot["licence"] == "CC-BY 4.0"
    assert snapshot["scenario"]["name"] == "forte"
    assert snapshot["peak_accumulation_mm"] == pytest.approx(25.0)
    assert len(snapshot["hourly"]["time"]) == len(snapshot["hourly"]["precipitation_mm"])
    assert len(snapshot["hourly"]["accumulation_mm"]) == len(snapshot["hourly"]["time"])
    assert snapshot["generated_at"].endswith("Z")


def test_snapshot_carries_the_provenance_of_the_thresholds(config):
    """O site precisa poder citar a fonte do corte, não só mostrar o número."""
    snapshot = build_snapshot(series_of([1.0] * 5), config.risk_scenarios, config.forecast)
    assert snapshot["justification_pending"] is False
    assert "INMET" in snapshot["scenario_source"]


def test_snapshot_reports_both_peaks_and_which_one_triggered(config):
    """Uma hora de 25 mm, sem volume no dia: o cenário vem da intensidade."""
    snapshot = build_snapshot(
        series_of([0.0] * 5 + [25.0] + [0.0] * 30), config.risk_scenarios, config.forecast
    )
    assert snapshot["peak_hourly_mm"] == pytest.approx(25.0)
    assert snapshot["scenario"]["name"] == "forte"
    assert snapshot["scenario"]["triggered_by"] == "intensidade horária"


def test_a_long_drizzle_is_triggered_by_the_daily_criterion(config):
    """60 mm em 24 h sem nenhuma hora forte: o cenário vem do volume."""
    snapshot = build_snapshot(
        series_of([2.5] * 24), config.risk_scenarios, config.forecast
    )
    assert snapshot["peak_hourly_mm"] == pytest.approx(2.5)
    assert snapshot["peak_daily_mm"] == pytest.approx(60.0)
    assert snapshot["scenario"]["name"] == "forte"
    assert snapshot["scenario"]["triggered_by"] == "acumulado em 24 h"


def test_write_snapshot_creates_parents_and_valid_json(config, tmp_path):
    snapshot = build_snapshot(series_of([3.0] * 12), config.risk_scenarios, config.forecast)
    target = tmp_path / "nested" / "forecast_snapshot.json"
    written = write_snapshot(snapshot, target)

    assert written == target
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert reloaded["scenario"]["name"] == snapshot["scenario"]["name"]

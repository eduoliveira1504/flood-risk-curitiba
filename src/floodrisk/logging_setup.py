"""Configuração de logging do pipeline."""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"

# Bibliotecas geoespaciais e de rede são barulhentas em DEBUG.
_NOISY = ("rasterio", "fiona", "urllib3", "botocore", "matplotlib", "PIL", "asyncio")


def setup_logging(
    level: str | int | None = None,
    log_dir: Path | None = None,
    stage: str = "pipeline",
) -> logging.Logger:
    """Instala handlers de console e (opcionalmente) de arquivo.

    Idempotente: chamar duas vezes não duplica saída.
    """
    resolved = level or os.environ.get("FLOODRISK_LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(resolved)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        file_handler = logging.FileHandler(log_dir / f"{stage}_{stamp}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

    for name in _NOISY:
        logging.getLogger(name).setLevel(max(resolved, logging.WARNING))

    return logging.getLogger("floodrisk")

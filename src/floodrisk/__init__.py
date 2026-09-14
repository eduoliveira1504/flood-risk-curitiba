"""flood-risk-curitiba — mapeamento de suscetibilidade a alagamentos urbanos."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, ConfigError, load_config

__all__ = ["Config", "ConfigError", "__version__", "load_config"]

"""Derivação de features a partir dos dados brutos.

Fase 2: o que os módulos daqui produzem já é insumo direto de treino. Nada aqui
fala com a rede — tudo lê de ``data/raw``/``data/interim`` e escreve em
``data/interim``/``data/processed``.
"""

from __future__ import annotations

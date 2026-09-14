"""Aquisição de dados externos.

Cada módulo aqui fala com uma fonte e materializa o resultado em ``data/raw/``.
Nenhum deles transforma dado além do necessário para persistir: limpeza e
derivação são responsabilidade da Fase 2.
"""

from __future__ import annotations

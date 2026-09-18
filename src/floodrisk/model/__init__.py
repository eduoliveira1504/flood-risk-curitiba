"""Componentes de Deep Learning: dataset, perda e laço de treino da U-Net.

Isolado do resto do pacote de propósito. Todo o pipeline das Fases 0–2 roda com
``pip install -e .`` puro; nada aqui é importado antes de o estágio ``train``
começar, então quem só quer os rasters não precisa de 3 GB de PyTorch.
"""

from __future__ import annotations

__all__: list[str] = []

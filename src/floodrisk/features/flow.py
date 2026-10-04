"""Talvegues derivados do relevo: para onde a água converge, segundo o DEM.

Existe para cobrir um buraco do cadastro. A hidrografia oficial desenha os rios
que se veem; os que a cidade enterrou — no centro de Curitiba, Ivo, Juvevê e
trechos do Belém correm em galeria — aparecem incompletos ou não aparecem. Mas
rio enterrado continua no fundo do mesmo vale, e é para lá que a água de chuva
escorre na superfície. O relevo guarda essa informação mesmo quando o cadastro
não guarda.

Três passos clássicos, sem dependência externa:

1. **Preenchimento de depressões** por *priority-flood* com épsilon (Barnes,
   Lehman e Mulla, 2014). Um DEM tem poços espúrios — ruído, prédio, viaduto —
   onde o escoamento calculado morreria. O algoritmo sobe a água a partir da
   borda e levanta cada poço até o nível do seu vertedouro; o épsilon dá uma
   inclinação mínima às áreas planas resultantes, para que tenham saída.
2. **Direção de fluxo D8**: cada célula drena para o vizinho de maior declive
   entre os oito.
3. **Área de contribuição**: quantas células drenam, por qualquer caminho, para
   cada uma. Célula com área de contribuição acima de um limiar é talvegue.

Roda na resolução NATIVA do DEM (30 m), não na grade de 10 m: reamostrar não
cria relevo, e a 10 m o cálculo custaria nove vezes mais para a mesma informação.
"""

from __future__ import annotations

import heapq

__all__ = ["FlowError", "contributing_area", "fill_depressions", "flow_receivers"]

#: Inclinação mínima imposta às áreas planas, em metros por célula. Pequena o
#: bastante para não alterar o relevo (mil células em fila somam 1 cm) e grande
#: o bastante para sobreviver ao arredondamento em float64.
EPSILON_M = 1e-5

_NEIGHBOURS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)  # fmt: skip


class FlowError(RuntimeError):
    """Entrada inválida na análise de escoamento."""


def fill_depressions(elevation):
    """Devolve o DEM sem poços, com saída garantida para toda célula.

    *Priority-flood*: a fila de prioridade começa com a borda e sempre processa
    a célula mais baixa já alcançada; um vizinho ainda não visitado que esteja
    abaixo dela só pode ser fundo de poço, e é levantado.
    """
    import numpy as np

    z = np.array(elevation, dtype="float64")
    if z.ndim != 2 or min(z.shape) < 3:
        raise FlowError(f"esperado array 2D de ao menos 3x3, veio {z.shape}")
    if not np.isfinite(z).all():
        raise FlowError("o DEM tem valores não finitos; preencha antes")

    rows, cols = z.shape
    closed = np.zeros(z.shape, dtype=bool)
    queue: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in (0, cols - 1):
            queue.append((z[r, c], r, c))
            closed[r, c] = True
    for c in range(1, cols - 1):
        for r in (0, rows - 1):
            queue.append((z[r, c], r, c))
            closed[r, c] = True
    heapq.heapify(queue)

    while queue:
        level, r, c = heapq.heappop(queue)
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols or closed[nr, nc]:
                continue
            closed[nr, nc] = True
            if z[nr, nc] <= level:
                z[nr, nc] = level + EPSILON_M
            heapq.heappush(queue, (z[nr, nc], nr, nc))
    return z


def flow_receivers(filled):
    """Índice (achatado) da célula para onde cada uma drena; -1 se não drena.

    D8: o vizinho de maior declive, com a diagonal dividida por √2 — sem isso o
    fluxo preferiria a diagonal só por ela ser mais longa.
    """
    import numpy as np

    z = np.asarray(filled, dtype="float64")
    rows, cols = z.shape
    padded = np.pad(z, 1, mode="constant", constant_values=np.inf)
    best_drop = np.zeros(z.shape)
    receiver = np.full(z.shape, -1, dtype="int64")
    index = np.arange(rows * cols).reshape(rows, cols)

    for dr, dc in _NEIGHBOURS:
        neighbour = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        drop = (z - neighbour) / (2**0.5 if dr and dc else 1.0)
        better = drop > best_drop
        best_drop = np.where(better, drop, best_drop)
        receiver = np.where(better, index + dr * cols + dc, receiver)
    return receiver


def contributing_area(elevation):
    """Número de células que drenam para cada célula, contando ela própria."""
    import numpy as np

    filled = fill_depressions(elevation)
    receiver = flow_receivers(filled).ravel()
    area = np.ones(receiver.size, dtype="float64")
    # Do ponto mais alto ao mais baixo: quando uma célula entrega sua área, tudo
    # que drena para ela já chegou.
    for cell in np.argsort(filled, axis=None)[::-1].tolist():
        target = receiver[cell]
        if target >= 0:
            area[target] += area[cell]
    return area.reshape(filled.shape)

"""Índice de suscetibilidade a alagamento, na grade zonal de 200 m.

O produto do TCC sai daqui. Duas camadas de 10 m entram — a impermeabilidade
predita pela U-Net e a declividade derivada do DEM — e sai uma grade de células
de 200 m com um índice contínuo e sua classificação.

**A regra física.** Superfície impermeável determina quanta chuva vira
escoamento em vez de infiltrar. Declividade determina se essa água fica ou
desce. Alagamento acontece onde as duas condições se encontram: muita água
gerada, pouca gravidade para levá-la embora. Impermeável em ladeira não alaga
ali — transfere o problema para baixo. Permeável e plano absorve. Por isso o
índice é o **produto**, e não a soma: soma deixaria uma célula íngreme e
totalmente impermeável empatar com uma plana e semipermeável, que são situações
fisicamente diferentes. No produto, um fator próximo de zero zera o resultado,
que é o comportamento correto.

**Por que percentil e não min–max.** A declividade máxima medida em Curitiba foi
54,33°, contra p95 de 14,11° — cauda finíssima, produzida por borda de vale num
DEM de 30 m reamostrado para 10 m. Normalizar por min–max entregaria a escala
inteira do índice a esses poucos pixels de ruído: metade da cidade cairia na
mesma faixa esmagada perto de zero. O percentil é insensível a isso por
construção, e tem a leitura direta de "esta célula é mais plana que X% da
cidade", que é o que um mapa municipal deve dizer.

**O que este estágio deliberadamente NÃO faz.** Não multiplica o índice por um
fator de chuva. Sem a base de ocorrências da Defesa Civil não existe como
calibrar a relação entre acumulado e área efetivamente alagada, e qualquer peso
inventado aqui ("crítica vale o dobro de forte") seria arbitrariedade disfarçada
de modelo — justamente o que a escolha dos patamares pelo INMET eliminou. O
índice é propriedade do TERRITÓRIO, estático; o cenário de chuva é exibido ao
lado como contexto. Calibrar essa relação é o trabalho futuro que a ausência de
dado impede hoje, e o documento deve dizer isso com essas palavras.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import artifacts
from ..config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "CLASS_LABELS",
    "SusceptibilityError",
    "build",
    "classify",
    "percentile_rank",
    "susceptibility_index",
]

#: Classes do mapa, do menor para o maior índice. Cinco classes por quintil:
#: número ímpar para que exista uma faixa "média" central legível, e quintis
#: para que cada classe tenha 20% do território por construção.
CLASS_LABELS = ("muito baixa", "baixa", "média", "alta", "muito alta")


class SusceptibilityError(RuntimeError):
    """Insumo ausente ou inconsistente na montagem do índice."""


def percentile_rank(values):
    """Posição de cada valor na distribuição, em (0, 1).

    Empates recebem o MESMO percentil — o correto, e o que um ``argsort`` puro
    erraria, distribuindo posições diferentes entre valores iguais.

    **Rank médio, não distribuição acumulada.** A tentação é usar a ECDF (fração
    de valores menores ou iguais), mas ela atribui exatamente 1,0 ao maior valor,
    e como a declividade entra invertida no índice, isso daria fator de
    planicidade ZERO à célula mais íngreme da cidade — zerando seu índice por
    completo, ainda que ela fosse 100% impermeável. Pior: numa área de
    declividade uniforme, TODAS as células empatariam em 1,0 e o mapa inteiro
    zeraria, apagando a informação de impermeabilidade.

    O rank médio (média das posições de início e fim do empate) não tem nenhum
    dos dois problemas: fica estritamente dentro de (0, 1) e, no caso uniforme,
    devolve 0,5 para todos — o que preserva o outro eixo do índice.
    """
    import numpy as np

    array = np.asarray(values, dtype="float64")
    if array.ndim != 1:
        raise SusceptibilityError(f"esperado vetor 1D, veio {array.ndim}D")
    if array.size == 0:
        raise SusceptibilityError("sem valores para ranquear")

    order = np.sort(array)
    below = np.searchsorted(order, array, side="left")
    through = np.searchsorted(order, array, side="right")
    return (below + through) / (2 * array.size)


def susceptibility_index(impervious, slope):
    """Índice contínuo em [0, 1] a partir da impermeabilidade e da declividade.

    ``impervious`` já é probabilidade média na célula, então entra direto. A
    declividade entra invertida — ``1 - percentil`` — porque plano é o que
    agrava: a célula mais plana da cidade recebe fator ~1 e a mais íngreme ~0.
    """
    import numpy as np

    impervious = np.asarray(impervious, dtype="float64")
    slope = np.asarray(slope, dtype="float64")
    if impervious.shape != slope.shape:
        raise SusceptibilityError(
            f"impermeabilidade {impervious.shape} e declividade {slope.shape} "
            "têm formas diferentes"
        )
    if impervious.size == 0:
        raise SusceptibilityError("nenhuma célula para calcular o índice")
    if (impervious < 0).any() or (impervious > 1).any():
        raise SusceptibilityError("a impermeabilidade precisa estar em [0, 1]")

    flatness = 1.0 - percentile_rank(slope)
    return impervious * flatness


def classify(index, labels=CLASS_LABELS):
    """Classifica o índice em faixas de igual população, por quantil.

    Corte por quantil e não por valor fixo: o índice é uma escala relativa —
    dizer "esta célula está entre os 20% mais suscetíveis de Curitiba" é
    afirmação verificável, enquanto "índice acima de 0,6 é risco alto" exigiria
    uma calibração contra alagamentos observados que o projeto ainda não tem.
    """
    import numpy as np

    array = np.asarray(index, dtype="float64")
    if array.size == 0:
        raise SusceptibilityError("nenhuma célula para classificar")
    if len(labels) < 2:
        raise SusceptibilityError("são necessárias ao menos duas classes")

    rank = percentile_rank(array)
    # `rank` chega a 1.0 no maior valor; sem o mínimo, ele cairia fora do vetor.
    position = np.minimum((rank * len(labels)).astype("int64"), len(labels) - 1)
    return position


def _zonal_means(config: Config, cells):
    """Média de cada raster de entrada dentro de cada célula da grade.

    A agregação é feita rasterizando o índice da célula e somando com
    ``bincount``, em vez de recortar raster por célula. São milhares de células;
    a versão ingênua abriria e mascararia o raster uma vez por célula.
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize

    probability_path = artifacts.impervious_probability(config)
    slope_path = artifacts.slope(config)
    missing = [p for p in (probability_path, slope_path) if not p.exists()]
    if missing:
        names = ", ".join(config.display_path(p) for p in missing)
        raise SusceptibilityError(
            f"insumo ausente: {names}. Rode 'infer' e 'build-terrain' antes."
        )

    with rasterio.open(probability_path) as source:
        probability = source.read(1)
        transform = source.transform
        shape = (source.height, source.width)
        nodata = source.nodata

    with rasterio.open(slope_path) as source:
        slope = source.read(1)
    if slope.shape != probability.shape:
        raise SusceptibilityError(
            "declividade e probabilidade estão em grades diferentes; "
            "refaça 'build-terrain' e 'infer'."
        )

    # 0 fica reservado para "fora de qualquer célula", então os índices começam
    # em 1 e o vetor de saída descarta a posição 0.
    zones = rasterize(
        ((cell, index + 1) for index, cell in enumerate(cells)),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )

    valid = np.isfinite(probability) & np.isfinite(slope)
    if nodata is not None:
        valid &= probability != nodata
    valid &= zones > 0

    flat_zones = zones[valid].astype("int64")
    count = np.bincount(flat_zones, minlength=len(cells) + 1)[1:]
    total_probability = np.bincount(
        flat_zones, weights=probability[valid].astype("float64"), minlength=len(cells) + 1
    )[1:]
    total_slope = np.bincount(
        flat_zones, weights=slope[valid].astype("float64"), minlength=len(cells) + 1
    )[1:]

    return count, total_probability, total_slope


def build(config: Config) -> Path:
    """Estágio ``susceptibility``: monta a grade zonal com o índice e as classes."""
    import geopandas as gpd
    import numpy as np

    from ..geo import aoi_geometry, grid_cells

    aoi = aoi_geometry(config, metric=True)
    cell_size = config.grid.cell_size_m
    cells = grid_cells(aoi, cell_size)
    if not cells:
        raise SusceptibilityError("a grade zonal não produziu nenhuma célula")
    logger.info(
        "grade de %.0f m: %d células sobre %.1f km²",
        cell_size,
        len(cells),
        aoi.area / 1e6,
    )

    count, total_probability, total_slope = _zonal_means(config, cells)

    # Célula de borda pode ter só uma nesga dentro do dado válido; a média ali
    # seria de meia dúzia de pixels e não representa a célula.
    expected = (cell_size / config.raster.resolution_m) ** 2
    enough = count >= config.grid.min_valid_fraction * expected
    dropped = int(len(cells) - enough.sum())
    if dropped:
        logger.info(
            "%d célula(s) descartada(s) por cobertura insuficiente (< %.0f%% de "
            "%.0f pixels)",
            dropped,
            100 * config.grid.min_valid_fraction,
            expected,
        )
    if not enough.any():
        raise SusceptibilityError(
            "nenhuma célula atingiu 'grid.min_valid_fraction'. A grade e os "
            "rasters estão na mesma projeção?"
        )

    kept = [cell for cell, keep in zip(cells, enough, strict=True) if keep]
    impervious = total_probability[enough] / count[enough]
    slope = total_slope[enough] / count[enough]

    index = susceptibility_index(impervious, slope)
    position = classify(index)

    frame = gpd.GeoDataFrame(
        {
            "cell_id": np.arange(len(kept)),
            "impervious_mean": np.round(impervious, 4),
            "slope_mean_deg": np.round(slope, 3),
            "flatness": np.round(1.0 - percentile_rank(slope), 4),
            "susceptibility": np.round(index, 4),
            "class_index": position,
            "class_label": [CLASS_LABELS[p] for p in position],
            "pixels": count[enough],
        },
        geometry=kept,
        crs=config.project.crs_metric,
    )

    destination = artifacts.susceptibility_cells(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(destination, layer="cells", driver="GPKG")
    logger.info("grade escrita em %s", config.display_path(destination))

    # O site lê GeoJSON, que obriga WGS84 pela própria especificação.
    web = artifacts.susceptibility_geojson(config)
    web.parent.mkdir(parents=True, exist_ok=True)
    frame.to_crs(config.project.crs_geo).to_file(web, driver="GeoJSON")
    logger.info("camada do site escrita em %s", config.display_path(web))

    _report(frame, config)
    return destination


def _report(frame, config: Config) -> None:
    """Resumo por classe, mais a leitura que interessa ao documento."""
    import numpy as np

    logger.info(
        "%-12s %8s %8s %14s %12s",
        "classe",
        "células",
        "%",
        "impermeável",
        "declividade",
    )
    total = len(frame)
    for position, label in enumerate(CLASS_LABELS):
        selection = frame[frame["class_index"] == position]
        if selection.empty:
            continue
        logger.info(
            "%-12s %8d %7.1f%% %13.1f%% %10.2f°",
            label,
            len(selection),
            100 * len(selection) / total,
            100 * selection["impervious_mean"].mean(),
            selection["slope_mean_deg"].mean(),
        )

    top = frame[frame["class_index"] == len(CLASS_LABELS) - 1]
    if not top.empty:
        area_km2 = len(top) * (config.grid.cell_size_m**2) / 1e6
        logger.info(
            "classe mais alta: %.1f km² — %.0f%% impermeável, %.2f° de "
            "declividade média",
            area_km2,
            100 * top["impervious_mean"].mean(),
            top["slope_mean_deg"].mean(),
        )

    # Se a impermeabilidade e a declividade fossem redundantes entre si, o
    # cruzamento não acrescentaria nada a um mapa de impermeabilidade puro.
    correlation = np.corrcoef(frame["impervious_mean"], frame["slope_mean_deg"])[0, 1]
    logger.info(
        "correlação entre impermeabilidade e declividade: %+.3f (r² = %.3f)",
        correlation,
        correlation**2,
    )

    _report_slope_contribution(frame)


def _report_slope_contribution(frame) -> None:
    """Quanto a declividade muda o mapa, em número de células reclassificadas.

    Responde à pergunta que a banca vai fazer: se o índice é dominado pela
    impermeabilidade, por que trazer o terreno? A resposta honesta não é "porque
    a física diz" — é a contagem de células que mudam de classe quando o segundo
    eixo entra. Se fosse perto de zero, o DEM e a declividade seriam trabalho
    decorativo e o índice deveria ser simplificado.
    """
    import numpy as np

    impervious_only = classify(frame["impervious_mean"].to_numpy())
    combined = frame["class_index"].to_numpy()

    shift = np.abs(impervious_only - combined)
    moved = float((shift > 0).mean())
    far = float((shift >= 2).mean())

    logger.info(
        "a declividade reclassifica %.1f%% das células em relação a um mapa de "
        "impermeabilidade pura (%.1f%% mudam duas classes ou mais)",
        100 * moved,
        100 * far,
    )

    promoted = frame[impervious_only < combined]
    demoted = frame[impervious_only > combined]
    if not promoted.empty:
        logger.info(
            "  sobem de classe por serem planas: %d células, %.2f° de média",
            len(promoted),
            promoted["slope_mean_deg"].mean(),
        )
    if not demoted.empty:
        logger.info(
            "  descem por serem íngremes: %d células, %.2f° de média",
            len(demoted),
            demoted["slope_mean_deg"].mean(),
        )

"""Índice de suscetibilidade a alagamento, na grade zonal de 200 m.

O produto do TCC sai daqui. Três camadas de 10 m entram — a impermeabilidade
predita pela U-Net, a declividade derivada do DEM e a distância ao curso d'água
mais próximo — e sai uma grade de células de 200 m com um índice contínuo e sua
classificação.

    índice = impermeabilidade × retenção
    retenção = √(planicidade × proximidade da drenagem)

**A regra física.** Alagamento precisa de duas coisas ao mesmo tempo: água
gerada e água que fica.

- *Geração* é a impermeabilidade: quanto da chuva vira escoamento em vez de
  infiltrar.
- *Retenção* é o terreno, e ele tem duas leituras que não se substituem. A
  **planicidade** (declividade invertida) diz se a água parada ali tem gravidade
  para sair. A **proximidade da drenagem** diz se o entorno despeja água ali: a
  rede de cursos d'água marca as linhas para onde o relevo converge. Um topo de
  divisor plano e um fundo de vale plano têm a mesma declividade, e só o segundo
  recebe a água dos outros.

**Por que produto e não soma.** Soma deixaria uma célula íngreme e totalmente
impermeável empatar com uma plana e semipermeável, que são situações
fisicamente diferentes. No produto, um fator próximo de zero derruba o
resultado, que é o comportamento correto.

**Por que a raiz na retenção.** Planicidade e proximidade são duas medidas do
MESMO fenômeno (a água fica), então dividem um único lugar na fórmula, pela
média geométrica. Sem a raiz, o terreno entraria duas vezes e pesaria o dobro
da superfície, sem que haja dado para justificar essa hierarquia. A média
geométrica (e não a aritmética) mantém a regra do produto dentro do terreno:
plano mas longe de qualquer drenagem, ou junto ao rio mas em encosta, não
recebem retenção alta.

**O que o índice mede — e o que não mede.** Mede ALAGAMENTO: acúmulo de água de
chuva em área urbanizada por escoamento que não dá vazão. Não mede INUNDAÇÃO
fluvial (o rio saindo da calha): margem de rio ocupada por parque tem
impermeabilidade perto de zero e fica em classe baixa, o que é correto para
alagamento e seria errado para inundação. O documento precisa usar o termo certo.

**Por que percentil e não min–max.** A declividade máxima medida em Curitiba foi
54,33°, contra p95 de 14,11° — cauda finíssima, produzida por borda de vale num
DEM de 30 m reamostrado para 10 m. Normalizar por min–max entregaria a escala
inteira do índice a esses poucos pixels de ruído. O percentil é insensível a
isso por construção, e tem a leitura direta de "esta célula é mais plana (ou
está mais perto da drenagem) que X% da cidade". Vale para os dois fatores de
terreno.

**O que este estágio deliberadamente NÃO faz.** Não multiplica o índice por um
fator de chuva. Sem a base de ocorrências da Defesa Civil não existe como
calibrar a relação entre acumulado e área efetivamente alagada, e qualquer peso
inventado aqui seria arbitrariedade disfarçada de modelo. O índice é propriedade
do TERRITÓRIO, estático; o cenário de chuva é exibido ao lado como contexto.
Pelo mesmo motivo a forma da fórmula não é calibrada: é uma regra física com
pesos iguais, e o estágio reporta quanto o mapa muda sob formulações
alternativas para que essa escolha seja discutível com número.
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
    "retention_factor",
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


def retention_factor(slope, drainage_distance):
    """Fator de retenção em (0, 1): média geométrica de planicidade e proximidade.

    Os dois entram invertidos — ``1 - percentil`` — porque é o plano e o próximo
    da drenagem que agravam.
    """
    import numpy as np

    flatness = 1.0 - percentile_rank(slope)
    proximity = 1.0 - percentile_rank(drainage_distance)
    return np.sqrt(flatness * proximity)


def susceptibility_index(impervious, slope, drainage_distance):
    """Índice contínuo em [0, 1]: impermeabilidade × retenção do terreno.

    ``impervious`` já é probabilidade média na célula, então entra direto.
    """
    import numpy as np

    impervious = np.asarray(impervious, dtype="float64")
    slope = np.asarray(slope, dtype="float64")
    drainage_distance = np.asarray(drainage_distance, dtype="float64")
    if not (impervious.shape == slope.shape == drainage_distance.shape):
        raise SusceptibilityError(
            f"impermeabilidade {impervious.shape}, declividade {slope.shape} e "
            f"distância à drenagem {drainage_distance.shape} têm formas diferentes"
        )
    if impervious.size == 0:
        raise SusceptibilityError("nenhuma célula para calcular o índice")
    if (impervious < 0).any() or (impervious > 1).any():
        raise SusceptibilityError("a impermeabilidade precisa estar em [0, 1]")
    if (drainage_distance < 0).any():
        raise SusceptibilityError("a distância à drenagem não pode ser negativa")

    return impervious * retention_factor(slope, drainage_distance)


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
    distance_path = artifacts.drainage_distance(config)
    missing = [
        p for p in (probability_path, slope_path, distance_path) if not p.exists()
    ]
    if missing:
        names = ", ".join(config.display_path(p) for p in missing)
        raise SusceptibilityError(
            f"insumo ausente: {names}. Rode 'infer', 'build-terrain' e "
            "'build-drainage' antes."
        )

    with rasterio.open(probability_path) as source:
        probability = source.read(1)
        transform = source.transform
        shape = (source.height, source.width)
        nodata = source.nodata

    layers = {}
    for name, path in (("slope", slope_path), ("distance", distance_path)):
        with rasterio.open(path) as source:
            layers[name] = source.read(1)
        if layers[name].shape != probability.shape:
            raise SusceptibilityError(
                f"{config.display_path(path)} está em grade diferente da "
                "probabilidade; refaça o estágio que o produz."
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

    valid = np.isfinite(probability) & (zones > 0)
    for layer in layers.values():
        valid &= np.isfinite(layer)
    if nodata is not None:
        valid &= probability != nodata

    flat_zones = zones[valid].astype("int64")
    size = len(cells) + 1
    count = np.bincount(flat_zones, minlength=size)[1:]

    def total(layer):
        return np.bincount(
            flat_zones, weights=layer[valid].astype("float64"), minlength=size
        )[1:]

    return count, total(probability), total(layers["slope"]), total(layers["distance"])


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

    count, total_probability, total_slope, total_distance = _zonal_means(config, cells)

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
    distance = total_distance[enough] / count[enough]

    index = susceptibility_index(impervious, slope, distance)
    position = classify(index)

    frame = gpd.GeoDataFrame(
        {
            "cell_id": np.arange(len(kept)),
            "impervious_mean": np.round(impervious, 4),
            "slope_mean_deg": np.round(slope, 3),
            "drainage_dist_m": np.round(distance, 1),
            "flatness": np.round(1.0 - percentile_rank(slope), 4),
            "proximity": np.round(1.0 - percentile_rank(distance), 4),
            "retention": np.round(retention_factor(slope, distance), 4),
            "susceptibility": np.round(index, 4),
            "class_index": position,
            "class_label": [CLASS_LABELS[p] for p in position],
            "pixels": count[enough],
        },
        geometry=kept,
        crs=config.project.crs_metric,
    )

    frame["basin"] = _zone_of(
        frame, artifacts.basins(config), config.drainage.basins_name_field, "bacia", config
    )
    frame["neighbourhood"] = _zone_of(
        frame,
        artifacts.neighbourhoods(config),
        "name",
        "bairro",
        config,
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


def _zone_of(frame, path: Path, field: str, label: str, config: Config):
    """Nome da zona (bacia, bairro) que contém o centro de cada célula.

    Atributo de agregação, não fator do índice: permite dizer "a bacia do Belém
    tem X% das células na faixa mais alta". Sem o arquivo a coluna sai vazia e
    o resto do pipeline segue igual.
    """
    import geopandas as gpd

    if not path.exists():
        logger.warning(
            "%s ausente (%s): a grade fica sem esse resumo",
            label,
            config.display_path(path),
        )
        return None

    zones = gpd.read_file(path)
    if field not in zones.columns:
        raise SusceptibilityError(
            f"{config.display_path(path)} não tem o campo '{field}'"
        )
    if zones.crs is None:
        raise SusceptibilityError(f"{config.display_path(path)} está sem CRS declarado")
    zones = zones.to_crs(frame.crs)[[field, "geometry"]]

    centres = gpd.GeoDataFrame(geometry=frame.geometry.centroid, crs=frame.crs)
    joined = gpd.sjoin(centres, zones, how="left", predicate="within")
    # Polígonos vizinhos podem se sobrepor num fio; fica o primeiro.
    joined = joined[~joined.index.duplicated(keep="first")]
    names = joined[field].reindex(frame.index)

    summary = (
        frame.assign(zone=names)
        .groupby("zone")
        .agg(
            cells=("cell_id", "size"),
            index=("susceptibility", "mean"),
            top=("class_index", lambda s: (s == len(CLASS_LABELS) - 1).mean()),
        )
        .sort_values("top", ascending=False)
    )
    logger.info(
        "por %s (células · índice médio · %% na faixa mais alta) — %d no total, "
        "os primeiros:",
        label,
        len(summary),
    )
    for name, row in summary.head(10).iterrows():
        logger.info(
            "  %-28s %6d  %.3f  %5.1f%%", name, row["cells"], row["index"], 100 * row["top"]
        )
    return names.where(names.notna(), None)


def _report(frame, config: Config) -> None:
    """Resumo por classe, mais a leitura que interessa ao documento."""
    import numpy as np

    logger.info(
        "%-12s %8s %7s %12s %12s %12s %14s",
        "classe",
        "células",
        "%",
        "impermeável",
        "declividade",
        "dist. rio",
        "a ≤100 m rio",
    )
    total = len(frame)
    for position, label in enumerate(CLASS_LABELS):
        selection = frame[frame["class_index"] == position]
        if selection.empty:
            continue
        logger.info(
            "%-12s %8d %6.1f%% %11.1f%% %11.2f° %10.0f m %13.1f%%",
            label,
            len(selection),
            100 * len(selection) / total,
            100 * selection["impervious_mean"].mean(),
            selection["slope_mean_deg"].mean(),
            selection["drainage_dist_m"].median(),
            100 * (selection["drainage_dist_m"] <= 100).mean(),
        )

    top = frame[frame["class_index"] == len(CLASS_LABELS) - 1]
    if not top.empty:
        area_km2 = len(top) * (config.grid.cell_size_m**2) / 1e6
        logger.info(
            "classe mais alta: %.1f km² — %.0f%% impermeável, %.2f° de "
            "declividade média, %.0f m de distância mediana à drenagem",
            area_km2,
            100 * top["impervious_mean"].mean(),
            top["slope_mean_deg"].mean(),
            top["drainage_dist_m"].median(),
        )

    # Se os fatores fossem redundantes entre si, o cruzamento não acrescentaria
    # nada a um mapa de impermeabilidade puro.
    columns = {
        "impermeabilidade": "impervious_mean",
        "declividade": "slope_mean_deg",
        "distância à drenagem": "drainage_dist_m",
    }
    names = list(columns)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            r = np.corrcoef(frame[columns[left]], frame[columns[right]])[0, 1]
            logger.info("correlação %s × %s: %+.3f (r² = %.3f)", left, right, r, r**2)

    _report_factor_contribution(frame)
    _report_sensitivity(frame)


def _class_shift(reference, combined) -> tuple[float, float]:
    import numpy as np

    shift = np.abs(np.asarray(reference) - np.asarray(combined))
    return float((shift > 0).mean()), float((shift >= 2).mean())


def _report_factor_contribution(frame) -> None:
    """Quanto cada fator de terreno muda o mapa, em células reclassificadas.

    Responde à pergunta que a banca vai fazer: se o índice é dominado pela
    impermeabilidade, por que trazer o terreno? A resposta honesta não é "porque
    a física diz" — é a contagem de células que mudam de classe quando o fator
    entra. Se fosse perto de zero, o fator seria trabalho decorativo.
    """
    impervious = frame["impervious_mean"].to_numpy()
    flatness = frame["flatness"].to_numpy()
    combined = frame["class_index"].to_numpy()

    impervious_only = classify(impervious)
    without_drainage = classify(impervious * flatness)

    moved, far = _class_shift(impervious_only, combined)
    logger.info(
        "o terreno (declividade + drenagem) reclassifica %.1f%% das células em "
        "relação a um mapa de impermeabilidade pura (%.1f%% mudam duas classes "
        "ou mais)",
        100 * moved,
        100 * far,
    )
    moved, far = _class_shift(without_drainage, combined)
    logger.info(
        "a proximidade da drenagem reclassifica %.1f%% das células em relação ao "
        "índice só com declividade (%.1f%% mudam duas classes ou mais)",
        100 * moved,
        100 * far,
    )

    promoted = frame[without_drainage < combined]
    demoted = frame[without_drainage > combined]
    if not promoted.empty:
        logger.info(
            "  sobem de classe por estarem junto à drenagem: %d células, %.0f m "
            "de distância mediana",
            len(promoted),
            promoted["drainage_dist_m"].median(),
        )
    if not demoted.empty:
        logger.info(
            "  descem por estarem longe dela: %d células, %.0f m de distância "
            "mediana",
            len(demoted),
            demoted["drainage_dist_m"].median(),
        )


def _report_sensitivity(frame) -> None:
    """Quanto o mapa depende da FORMA escolhida para combinar os fatores.

    Sem ocorrências observadas não há como calibrar a fórmula; o que dá para
    fazer é medir o quanto a classificação muda sob alternativas razoáveis. As
    células que ficam na classe mais alta em todas elas são o resultado que não
    depende da escolha.
    """
    import numpy as np

    impervious = frame["impervious_mean"].to_numpy()
    flatness = frame["flatness"].to_numpy()
    proximity = frame["proximity"].to_numpy()
    combined = frame["class_index"].to_numpy()
    top_class = len(CLASS_LABELS) - 1

    alternatives = {
        "produto simples (imp × plan × prox)": impervious * flatness * proximity,
        "média aritmética (imp × (plan + prox) / 2)": impervious
        * (flatness + proximity)
        / 2,
    }
    stable = combined == top_class
    logger.info("sensibilidade à forma da fórmula:")
    for name, index in alternatives.items():
        position = classify(index)
        moved, far = _class_shift(position, combined)
        stable &= position == top_class
        logger.info(
            "  %-44s %5.1f%% das células em outra classe (%.1f%% a duas ou mais)",
            name,
            100 * moved,
            100 * far,
        )
    logger.info(
        "  células na classe mais alta nas três formulações: %d de %d (%.0f%%)",
        int(stable.sum()),
        int((combined == top_class).sum()),
        100 * stable.sum() / max(int((combined == top_class).sum()), 1),
    )
    logger.debug("alternativas avaliadas: %s", np.array(list(alternatives)))

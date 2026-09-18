from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from floodrisk.config import Config, ConfigError, find_repo_root, load_config
from floodrisk.features.susceptibility import (
    CLASS_LABELS,
    SusceptibilityError,
    classify,
    percentile_rank,
    susceptibility_index,
)


@pytest.fixture
def config():
    return load_config()


def raw_config():
    with (find_repo_root() / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_config(data):
    return Config.from_dict(data, root=find_repo_root())


# --------------------------------------------------------------------------- #
# Percentil
# --------------------------------------------------------------------------- #


def test_the_rank_is_monotonic():
    ranks = percentile_rank([5.0, 1.0, 3.0, 9.0])
    assert ranks[1] < ranks[2] < ranks[0] < ranks[3]


def test_the_rank_stays_strictly_inside_the_unit_interval():
    """Nem 0 nem 1: o extremo em 1 zeraria o índice da célula mais íngreme."""
    ranks = percentile_rank(np.random.default_rng(0).normal(5, 2, 200))
    assert ranks.min() > 0 and ranks.max() < 1.0


def test_ties_receive_the_same_rank():
    """argsort puro daria posições diferentes a valores iguais — e duas células
    igualmente planas não podem receber suscetibilidades diferentes."""
    ranks = percentile_rank([2.0, 2.0, 2.0, 9.0])
    assert ranks[0] == ranks[1] == ranks[2]


def test_the_largest_value_does_not_reach_the_top_of_the_scale():
    """Se chegasse a 1,0, a célula mais íngreme teria planicidade zero e índice
    zero — mesmo sendo inteiramente impermeável."""
    assert percentile_rank([1.0, 2.0, 3.0])[-1] < 1.0


def test_a_uniform_distribution_collapses_to_the_middle():
    """Se toda a cidade tivesse a mesma declividade, a ECDF daria 1,0 a todas e
    o mapa inteiro zeraria, apagando a impermeabilidade. O rank médio dá 0,5."""
    assert np.allclose(percentile_rank([3.0, 3.0, 3.0, 3.0]), 0.5)


def test_the_rank_ignores_the_scale_of_the_data():
    """É o que torna o índice imune à cauda de 54° da declividade."""
    base = [1.0, 2.0, 3.0, 4.0]
    outlier = [1.0, 2.0, 3.0, 10_000.0]
    assert np.allclose(percentile_rank(base), percentile_rank(outlier))


def test_an_outlier_does_not_crush_the_rest():
    """A razão de existir do percentil, medida contra a alternativa.

    Com min–max, a declividade de 54,33° medida em Curitiba (contra p95 de
    14,11°) espremeria quase toda a cidade no fundo da escala. Aqui os valores
    comuns continuam espalhados pela faixa toda.
    """
    values = np.array([1.0, 2.0, 3.0, 4.0, 999.0])
    ranks = percentile_rank(values)
    min_max = (values - values.min()) / (values.max() - values.min())

    # Os quatro valores comuns ocupam quase toda a escala no percentil...
    assert ranks[3] - ranks[0] > 0.5
    # ...e praticamente nada dela no min-max, esmagados pelo outlier.
    assert min_max[3] - min_max[0] < 0.01


def test_an_empty_input_is_rejected():
    with pytest.raises(SusceptibilityError, match="sem valores"):
        percentile_rank([])


def test_a_two_dimensional_input_is_rejected():
    with pytest.raises(SusceptibilityError, match="1D"):
        percentile_rank(np.zeros((3, 3)))


# --------------------------------------------------------------------------- #
# Índice
# --------------------------------------------------------------------------- #


def test_flat_and_sealed_beats_steep_and_sealed():
    """A regra física do projeto: impermeável em ladeira transfere, não acumula."""
    index = susceptibility_index([0.9, 0.9], [1.0, 30.0])
    assert index[0] > index[1]


def test_sealed_beats_permeable_at_the_same_slope():
    index = susceptibility_index([0.9, 0.1, 0.5], [5.0, 5.0, 5.0])
    assert index[0] > index[2] > index[1]


def test_a_permeable_cell_scores_near_zero_however_flat():
    index = susceptibility_index([0.0, 0.8], [0.1, 20.0])
    assert index[0] == pytest.approx(0.0)


def test_the_index_is_a_product_not_a_sum():
    """Numa soma, íngreme-e-impermeável empataria com plano-e-semipermeável —
    situações fisicamente distintas. No produto, um fator baixo domina."""
    impervious = [1.0, 0.5, 0.5, 0.0]
    slope = [30.0, 10.0, 10.0, 1.0]
    index = susceptibility_index(impervious, slope)
    steep_sealed = index[0]
    assert steep_sealed < max(index)


def test_the_index_stays_inside_the_unit_interval():
    rng = np.random.default_rng(1)
    index = susceptibility_index(rng.random(500), rng.gamma(2, 3, 500))
    assert index.min() >= 0 and index.max() <= 1


def test_mismatched_shapes_are_rejected():
    with pytest.raises(SusceptibilityError, match="formas diferentes"):
        susceptibility_index([0.5, 0.5], [1.0])


def test_a_probability_outside_the_unit_interval_is_rejected():
    with pytest.raises(SusceptibilityError, match=r"\[0, 1\]"):
        susceptibility_index([1.5], [1.0])


def test_no_cells_is_rejected():
    with pytest.raises(SusceptibilityError, match="nenhuma célula"):
        susceptibility_index([], [])


# --------------------------------------------------------------------------- #
# Classificação
# --------------------------------------------------------------------------- #


def test_every_class_index_is_addressable():
    """Índice fora da lista de rótulos viraria IndexError na hora de nomear."""
    position = classify(np.random.default_rng(2).random(1000))
    assert position.min() >= 0
    assert position.max() == len(CLASS_LABELS) - 1


def test_the_highest_value_lands_in_the_top_class():
    position = classify([0.1, 0.2, 0.3, 0.9])
    assert position[-1] == len(CLASS_LABELS) - 1


def test_the_classes_split_the_territory_evenly():
    """Quintis: cada classe fica com ~20% das células por construção."""
    position = classify(np.random.default_rng(3).random(5000))
    counts = np.bincount(position, minlength=len(CLASS_LABELS))
    assert all(abs(count / 5000 - 0.2) < 0.02 for count in counts)


def test_the_classification_preserves_the_ordering():
    values = np.random.default_rng(4).random(300)
    position = classify(values)
    for left in range(len(values)):
        for right in range(len(values)):
            if values[left] < values[right]:
                assert position[left] <= position[right]
            break


def test_a_single_cell_is_classified_without_crashing():
    """Uma célula sozinha não é "muito alta" nem "muito baixa" — não há
    distribuição contra a qual compará-la, então ela cai no meio."""
    assert classify([0.4]).tolist() == [len(CLASS_LABELS) // 2]


def test_fewer_than_two_classes_is_rejected():
    with pytest.raises(SusceptibilityError, match="duas classes"):
        classify([0.1, 0.2], labels=("única",))


def test_classifying_nothing_is_rejected():
    with pytest.raises(SusceptibilityError, match="nenhuma célula"):
        classify([])


# --------------------------------------------------------------------------- #
# Configuração da grade
# --------------------------------------------------------------------------- #


def test_the_cell_holds_a_whole_number_of_pixels(config):
    """200 m sobre pixel de 10 m dá 20 x 20 exatos; fração deixaria borda órfã."""
    ratio = config.grid.cell_size_m / config.raster.resolution_m
    assert ratio == int(ratio)


def test_a_non_positive_cell_is_rejected():
    data = copy.deepcopy(raw_config())
    data["grid"]["cell_size_m"] = 0
    with pytest.raises(ConfigError, match="cell_size_m"):
        build_config(data)


def test_a_coverage_fraction_outside_the_range_is_rejected():
    data = copy.deepcopy(raw_config())
    data["grid"]["min_valid_fraction"] = 0.0
    with pytest.raises(ConfigError, match="min_valid_fraction"):
        build_config(data)


# --------------------------------------------------------------------------- #
# Contribuição do segundo eixo
# --------------------------------------------------------------------------- #


def test_slope_actually_moves_cells_between_classes():
    """Se a declividade não reclassificasse nada, o DEM seria trabalho decorativo
    e o índice deveria ser simplificado para impermeabilidade pura."""
    rng = np.random.default_rng(5)
    impervious = rng.random(2000)
    slope = rng.gamma(2, 3, 2000)

    combined = classify(susceptibility_index(impervious, slope))
    impervious_only = classify(impervious)
    assert (combined != impervious_only).mean() > 0.1


def test_with_uniform_slope_the_two_maps_agree():
    """Terreno sem variação não pode reordenar nada — se reordenasse, o fator de
    planicidade estaria inventando diferença onde não há."""
    impervious = np.linspace(0.01, 0.99, 500)
    slope = np.full(500, 4.2)

    combined = classify(susceptibility_index(impervious, slope))
    assert np.array_equal(combined, classify(impervious))


def test_a_flat_cell_never_ranks_below_a_steeper_twin():
    """Mesma impermeabilidade, declividades diferentes: a plana vem na frente."""
    index = susceptibility_index([0.6, 0.6, 0.6], [1.0, 8.0, 25.0])
    assert index[0] > index[1] > index[2]

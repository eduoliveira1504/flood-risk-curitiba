from __future__ import annotations

import copy
import itertools

import pytest
import yaml
from shapely.geometry import box

from floodrisk.config import Config, ConfigError, find_repo_root, load_config
from floodrisk.features.dataset import (
    DatasetError,
    Patch,
    assign_blocks,
    block_id,
    blocks_touched,
    choose_seed,
    prevalence_spread,
    resolve_splits,
)


@pytest.fixture
def config():
    return load_config()


def raw_config():
    with (find_repo_root() / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_config(data):
    return Config.from_dict(data, root=find_repo_root())


def patch(west, south, size=1280.0, **kwargs):
    """Patch sintético posicionado por sua pegada métrica."""
    return Patch(
        col_off=int(west // 10),
        row_off=int(south // 10),
        width=int(size // 10),
        height=int(size // 10),
        bounds=(west, south, west + size, south + size),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Identificação de bloco
# --------------------------------------------------------------------------- #


def test_point_falls_in_its_block():
    assert block_id(0.0, 0.0, 1000.0) == (0, 0)
    assert block_id(1500.0, 2500.0, 1000.0) == (1, 2)


def test_block_uses_the_floor_not_truncation():
    """int() truncaria -0.3 para 0 e colaria dois blocos vizinhos num só."""
    assert block_id(-300.0, -300.0, 1000.0) == (-1, -1)


def test_block_boundary_belongs_to_the_upper_block():
    assert block_id(1000.0, 1000.0, 1000.0) == (1, 1)


def test_non_positive_block_size_is_rejected():
    with pytest.raises(DatasetError, match="block_size_m"):
        block_id(0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Blocos tocados pela pegada
# --------------------------------------------------------------------------- #


def test_patch_inside_one_block_touches_only_it():
    assert blocks_touched((100.0, 100.0, 900.0, 900.0), 1000.0) == {(0, 0)}


def test_patch_crossing_a_boundary_touches_both():
    assert blocks_touched((900.0, 100.0, 1100.0, 900.0), 1000.0) == {(0, 0), (1, 0)}


def test_patch_over_a_corner_touches_four():
    assert len(blocks_touched((900.0, 900.0, 1100.0, 1100.0), 1000.0)) == 4


def test_edge_exactly_on_the_boundary_does_not_touch_the_next_block():
    """Encostar na divisa divide uma linha de área zero — e nenhum pixel."""
    assert blocks_touched((0.0, 0.0, 1000.0, 1000.0), 1000.0) == {(0, 0)}


def test_patch_larger_than_the_block_touches_several():
    assert len(blocks_touched((0.0, 0.0, 2500.0, 2500.0), 1000.0)) == 9


def test_degenerate_footprint_is_rejected():
    with pytest.raises(DatasetError, match="degenerada"):
        blocks_touched((100.0, 100.0, 100.0, 900.0), 1000.0)


# --------------------------------------------------------------------------- #
# Sorteio dos blocos
# --------------------------------------------------------------------------- #


FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def uniform_weights(n):
    return {(col, 0): 1 for col in range(n)}


def test_every_block_gets_a_split():
    assignment = assign_blocks(uniform_weights(40), FRACTIONS, seed=42)
    assert len(assignment) == 40
    assert set(assignment.values()) == {"train", "val", "test"}


def test_proportions_follow_the_requested_fractions():
    assignment = assign_blocks(uniform_weights(100), FRACTIONS, seed=42)
    counts = {name: sum(1 for v in assignment.values() if v == name) for name in FRACTIONS}
    for name, fraction in FRACTIONS.items():
        assert abs(counts[name] / 100 - fraction) < 0.03


def test_fractions_are_counted_in_patches_not_blocks():
    """Bloco de borda tem menos patch que bloco de miolo; a conta é por patch."""
    weights = {(0, 0): 100, (1, 0): 1, (2, 0): 1, (3, 0): 1}
    assignment = assign_blocks(weights, {"train": 0.5, "test": 0.5}, seed=0)
    # O bloco pesado sozinho já passa de 50%; os leves têm de ir todos para o outro.
    heavy = assignment[(0, 0)]
    assert all(assignment[b] != heavy for b in [(1, 0), (2, 0), (3, 0)])


def test_the_same_seed_gives_the_same_split():
    a = assign_blocks(uniform_weights(50), FRACTIONS, seed=7)
    b = assign_blocks(uniform_weights(50), FRACTIONS, seed=7)
    assert a == b


def test_a_different_seed_gives_a_different_split():
    a = assign_blocks(uniform_weights(50), FRACTIONS, seed=7)
    b = assign_blocks(uniform_weights(50), FRACTIONS, seed=8)
    assert a != b


def test_empty_universe_is_rejected():
    with pytest.raises(DatasetError, match="nenhum bloco"):
        assign_blocks({}, FRACTIONS, seed=0)


def test_all_zero_fractions_are_rejected():
    with pytest.raises(DatasetError, match="zero"):
        assign_blocks(uniform_weights(5), {"train": 0.0}, seed=0)


def test_a_zero_fraction_split_receives_nothing():
    assignment = assign_blocks(
        uniform_weights(20), {"train": 1.0, "val": 0.0}, seed=1
    )
    assert set(assignment.values()) == {"train"}


# --------------------------------------------------------------------------- #
# A garantia que o estágio inteiro existe para dar
# --------------------------------------------------------------------------- #


def grid_of_patches(cols, rows, stride=1120.0, size=1280.0):
    return [
        patch(col * stride, row * stride, size=size)
        for row in range(rows)
        for col in range(cols)
    ]


def test_no_two_patches_of_different_splits_ever_overlap():
    """A propriedade central: um pixel não pode estar em dois conjuntos.

    Se este teste falhar, toda métrica de teste do projeto está inflada.
    """
    result = resolve_splits(grid_of_patches(12, 12), 5120.0, FRACTIONS, seed=42)
    for (left, left_split), (right, right_split) in itertools.combinations(result.kept, 2):
        if left_split == right_split:
            continue
        assert box(*left.bounds).intersection(box(*right.bounds)).area == 0


def test_a_kept_patch_touches_only_blocks_of_its_own_split():
    result = resolve_splits(grid_of_patches(10, 10), 5120.0, FRACTIONS, seed=3)
    for item, split in result.kept:
        assert all(
            result.assignment[block] == split
            for block in blocks_touched(item.bounds, 5120.0)
        )


def test_patches_crossing_a_split_boundary_are_dropped():
    result = resolve_splits(grid_of_patches(10, 10), 5120.0, FRACTIONS, seed=3)
    assert result.dropped
    assert len(result.kept) + len(result.dropped) == 100


def test_blocks_without_any_patch_centre_are_still_assigned():
    """Bloco de borda coberto só por pegadas reabriria o vazamento se ficasse de fora."""
    # Dois patches cujos centros caem em blocos distintos, ambos transbordando
    # para o bloco do meio, onde nenhum centro cai.
    items = [patch(0.0, 0.0, size=1280.0), patch(2500.0, 0.0, size=1280.0)]
    result = resolve_splits(items, 1000.0, {"train": 0.5, "test": 0.5}, seed=0)
    covered = set().union(*(blocks_touched(p.bounds, 1000.0) for p in items))
    assert covered <= set(result.assignment)


def test_most_patches_survive_with_the_configured_block_size(config):
    """A faixa morta é o preço da garantia — mas não pode comer o conjunto."""
    extent = config.raster.patch_size * config.raster.resolution_m
    stride = config.raster.stride * config.raster.resolution_m
    result = resolve_splits(
        grid_of_patches(20, 30, stride=stride, size=extent),
        config.split.block_size_m,
        config.split.fractions,
        seed=config.project.random_seed,
    )
    kept = len(result.kept) / (len(result.kept) + len(result.dropped))
    assert kept > 0.5, f"apenas {kept:.0%} dos patches sobreviveram ao split"


def test_every_split_receives_patches(config):
    extent = config.raster.patch_size * config.raster.resolution_m
    stride = config.raster.stride * config.raster.resolution_m
    result = resolve_splits(
        grid_of_patches(20, 30, stride=stride, size=extent),
        config.split.block_size_m,
        config.split.fractions,
        seed=config.project.random_seed,
    )
    assigned = {split for _, split in result.kept}
    assert assigned == set(config.split.fractions)


def test_resolve_is_deterministic():
    items = grid_of_patches(8, 8)
    a = resolve_splits(items, 5120.0, FRACTIONS, seed=42)
    b = resolve_splits(items, 5120.0, FRACTIONS, seed=42)
    assert [(p.name, s) for p, s in a.kept] == [(p.name, s) for p, s in b.kept]


def test_no_patches_is_rejected():
    with pytest.raises(DatasetError, match="nenhum patch"):
        resolve_splits([], 5120.0, FRACTIONS, seed=0)


# --------------------------------------------------------------------------- #
# Randomização restrita
# --------------------------------------------------------------------------- #


def gradient_patches(cols, rows, stride=1120.0, size=1280.0):
    """Grade onde a impermeabilidade cresce de oeste para leste.

    Imita o que o dado real faz: miolo urbano de um lado, periferia do outro. É
    justamente esse gradiente que faz um sorteio ingênuo desbalancear.
    """
    return [
        patch(
            col * stride,
            row * stride,
            size=size,
            positive_fraction=col / max(cols - 1, 1),
        )
        for row in range(rows)
        for col in range(cols)
    ]


def test_spread_is_zero_when_every_split_looks_alike():
    items = [patch(col * 1120.0, 0.0, positive_fraction=0.4) for col in range(30)]
    result = resolve_splits(items, 5120.0, FRACTIONS, seed=1)
    assert prevalence_spread(result) == pytest.approx(0.0, abs=1e-9)


def test_spread_is_reported_in_percentage_points():
    # Cada um no canto do seu bloco, para que nenhum caia na faixa morta.
    items = [
        patch(0.0, 0.0, positive_fraction=0.20),
        patch(20_480.0, 0.0, positive_fraction=0.50),
    ]
    result = resolve_splits(items, 5120.0, {"train": 0.5, "test": 0.5}, seed=0)
    assert prevalence_spread(result) == pytest.approx(30.0)


def test_a_single_split_has_no_spread():
    items = [patch(col * 1120.0, 0.0, positive_fraction=col / 10) for col in range(10)]
    result = resolve_splits(items, 5120.0, {"train": 1.0}, seed=0)
    assert prevalence_spread(result) == 0.0


def test_the_search_finds_a_balanced_seed():
    items = gradient_patches(24, 24)
    _, _, spread = choose_seed(items, 5120.0, FRACTIONS, 42, 5.0, 200)
    assert spread <= 5.0


def test_the_search_accepts_the_first_seed_that_passes_not_the_best():
    """Aceitar a melhor de todas seria escolher o resultado."""
    items = gradient_patches(24, 24)
    chosen, _, _ = choose_seed(items, 5120.0, FRACTIONS, 42, 5.0, 200)
    for candidate in range(42, chosen):
        earlier = resolve_splits(items, 5120.0, FRACTIONS, candidate)
        complete = {s for _, s in earlier.kept} == set(FRACTIONS)
        assert not (complete and prevalence_spread(earlier) <= 5.0)


def test_the_search_starts_at_the_configured_seed():
    items = gradient_patches(24, 24)
    chosen, _, _ = choose_seed(items, 5120.0, FRACTIONS, 42, 100.0, 200)
    assert chosen == 42


def test_the_search_is_deterministic():
    items = gradient_patches(20, 20)
    first = choose_seed(items, 5120.0, FRACTIONS, 7, 4.0, 150)
    second = choose_seed(items, 5120.0, FRACTIONS, 7, 4.0, 150)
    assert first[0] == second[0] and first[2] == pytest.approx(second[2])


def test_an_impossible_criterion_falls_back_to_the_least_bad():
    items = gradient_patches(20, 20)
    _, result, spread = choose_seed(items, 5120.0, FRACTIONS, 42, 0.0001, 25)
    assert spread > 0.0001
    assert {split for _, split in result.kept} == set(FRACTIONS)
    best = min(
        prevalence_spread(resolve_splits(items, 5120.0, FRACTIONS, s))
        for s in range(42, 67)
        if {x for _, x in resolve_splits(items, 5120.0, FRACTIONS, s).kept} == set(FRACTIONS)
    )
    assert spread == pytest.approx(best)


def test_a_seed_that_loses_a_whole_split_is_never_chosen():
    """Espalhamento baixo não vale nada se não há conjunto de teste."""
    items = gradient_patches(20, 20)
    _, result, _ = choose_seed(items, 5120.0, FRACTIONS, 42, 5.0, 200)
    assert {split for _, split in result.kept} == set(FRACTIONS)


def test_zero_attempts_is_rejected():
    with pytest.raises(DatasetError, match="max_seed_attempts"):
        choose_seed(gradient_patches(6, 6), 5120.0, FRACTIONS, 0, 5.0, 0)


# --------------------------------------------------------------------------- #
# Coerência entre seções da configuração
# --------------------------------------------------------------------------- #


def test_block_smaller_than_the_patch_is_rejected():
    data = copy.deepcopy(raw_config())
    data["split"]["block_size_m"] = 1000.0
    with pytest.raises(ConfigError, match="block_size_m"):
        build_config(data)


def test_block_exactly_the_patch_footprint_is_rejected():
    """Igual à pegada ainda faz quase todo patch cruzar divisa."""
    data = copy.deepcopy(raw_config())
    data["split"]["block_size_m"] = (
        data["raster"]["patch_size"] * data["raster"]["resolution_m"]
    )
    with pytest.raises(ConfigError, match="block_size_m"):
        build_config(data)


def test_growing_the_patch_without_growing_the_block_is_rejected():
    data = copy.deepcopy(raw_config())
    data["raster"]["patch_size"] = 512
    with pytest.raises(ConfigError, match="block_size_m"):
        build_config(data)


def test_the_shipped_configuration_is_coherent(config):
    extent = config.raster.patch_size * config.raster.resolution_m
    assert config.split.block_size_m >= 2 * extent


def test_patch_size_is_divisible_by_the_encoder_stride(config):
    """ResNet34 subamostra por 32; patch não múltiplo quebra o skip connection."""
    assert config.raster.patch_size % 32 == 0


def test_a_non_positive_spread_limit_is_rejected():
    data = copy.deepcopy(raw_config())
    data["split"]["max_prevalence_spread_pp"] = 0.0
    with pytest.raises(ConfigError, match="max_prevalence_spread_pp"):
        build_config(data)


def test_zero_seed_attempts_is_rejected_by_the_config():
    data = copy.deepcopy(raw_config())
    data["split"]["max_seed_attempts"] = 0
    with pytest.raises(ConfigError, match="max_seed_attempts"):
        build_config(data)


# --------------------------------------------------------------------------- #
# O estágio de ponta a ponta, sobre rasters sintéticos
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_repo(tmp_path):
    """Repositório de brinquedo com um mosaico e uma máscara de 10 x 10 km."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    (tmp_path / "configs").mkdir()
    data = raw_config()
    with (tmp_path / "configs" / "default.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)

    cfg = load_config(root=tmp_path)
    cfg.ensure_dirs()

    size = 1000
    # Canto dentro da bbox de Curitiba em SIRGAS 2000 / UTM 22S.
    transform = from_origin(660_000.0, 7_190_000.0, 10.0, 10.0)
    profile = {
        "driver": "GTiff",
        "width": size,
        "height": size,
        "transform": transform,
        "crs": cfg.project.crs_metric,
    }

    rng = np.random.default_rng(0)
    bands = rng.integers(200, 3000, size=(4, size, size), dtype="uint16")
    # Uma faixa sem dado, para exercitar 'min_valid_fraction'.
    bands[:, :, :60] = 0
    with rasterio.open(
        cfg.path("data_interim") / "s2_median.tif",
        "w",
        count=4,
        dtype="uint16",
        nodata=0,
        **profile,
    ) as dst:
        dst.write(bands)

    label = np.zeros((size, size), dtype="uint8")
    label[::3, :] = 1
    with rasterio.open(
        cfg.path("data_processed") / "impervious_mask.tif",
        "w",
        count=1,
        dtype="uint8",
        **profile,
    ) as dst:
        dst.write(label, 1)

    return cfg


def test_stage_writes_a_manifest_and_one_file_per_patch(fake_repo):
    import csv

    from floodrisk import artifacts
    from floodrisk.features import dataset

    manifest = dataset.build(fake_repo)
    assert manifest.exists()

    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert len(rows) == len(list(artifacts.patches_dir(fake_repo).rglob("*.npz")))


def test_stage_writes_image_mask_and_validity(fake_repo):
    import numpy as np

    from floodrisk import artifacts
    from floodrisk.features import dataset

    dataset.build(fake_repo)
    sample = next(iter(sorted(artifacts.patches_dir(fake_repo).rglob("*.npz"))))
    with np.load(sample) as payload:
        size = fake_repo.raster.patch_size
        assert payload["image"].shape == (fake_repo.model.in_channels, size, size)
        assert payload["mask"].shape == (size, size)
        assert payload["valid"].shape == (size, size)


def test_stage_drops_patches_over_the_nodata_stripe(fake_repo):
    """A faixa de 600 m sem dado tem de derrubar os patches da borda oeste."""
    import csv

    from floodrisk.features import dataset

    with dataset.build(fake_repo).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(float(row["valid_fraction"]) >= fake_repo.raster.min_valid_fraction
               for row in rows)
    assert min(int(row["col_off"]) for row in rows) > 0


def test_stage_is_idempotent(fake_repo):
    """Rodar duas vezes não pode deixar .npz de um split antigo para trás."""
    from floodrisk import artifacts
    from floodrisk.features import dataset

    dataset.build(fake_repo)
    first = sorted(p.relative_to(artifacts.patches_dir(fake_repo))
                   for p in artifacts.patches_dir(fake_repo).rglob("*.npz"))
    dataset.build(fake_repo)
    second = sorted(p.relative_to(artifacts.patches_dir(fake_repo))
                    for p in artifacts.patches_dir(fake_repo).rglob("*.npz"))
    assert first == second


def test_stage_refuses_a_mask_off_the_reference_grid(fake_repo):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from floodrisk import artifacts
    from floodrisk.features import dataset

    with rasterio.open(
        artifacts.impervious_mask(fake_repo),
        "w",
        driver="GTiff",
        width=500,
        height=500,
        count=1,
        dtype="uint8",
        crs=fake_repo.project.crs_metric,
        transform=from_origin(660_000.0, 7_190_000.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.zeros((500, 500), dtype="uint8"), 1)

    with pytest.raises(DatasetError, match="mesma grade"):
        dataset.build(fake_repo)


def test_stage_needs_its_inputs(tmp_path):
    from floodrisk.features import dataset

    (tmp_path / "configs").mkdir()
    with (tmp_path / "configs" / "default.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config(), handle)
    cfg = load_config(root=tmp_path)
    cfg.ensure_dirs()

    with pytest.raises(DatasetError, match="acquire-sentinel"):
        dataset.build(cfg)

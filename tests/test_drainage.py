from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from floodrisk.acquisition.drainage import (
    DrainageError,
    check_complete,
    parse_features,
    summarise,
)
from floodrisk.config import Config, ConfigError, find_repo_root, load_config
from floodrisk.features.drainage import DrainageDistanceError, distance_to_true
from floodrisk.pipeline import STAGES_BY_NAME


@pytest.fixture
def drainage():
    return load_config().drainage


def raw_config():
    with (find_repo_root() / "configs" / "default.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def feature(objectid=1, kind=4, covered=0, paths=None, name="Rio Belém"):
    return {
        "attributes": {
            "objectid": objectid,
            "nome": name,
            "tipotrechodrenagem": kind,
            "encoberto": covered,
            "geometriaaproximada": 0,
        },
        "geometry": {"paths": paths or [[[0.0, 0.0], [100.0, 0.0]]]},
    }


# --------------------------------------------------------------------------- #
# Leitura da resposta do serviço
# --------------------------------------------------------------------------- #


def test_a_watercourse_becomes_a_record(drainage):
    records = parse_features({"features": [feature()]}, drainage)
    assert len(records) == 1
    assert records[0]["name"] == "Rio Belém"
    assert records[0]["geometry"].length == pytest.approx(100.0)


def test_other_drainage_types_are_left_out(drainage):
    """Só curso d'água entra: o tipo pluvial, se um dia vier preenchido, é outra
    rede e mudaria o significado do fator sem ninguém decidir isso."""
    records = parse_features({"features": [feature(kind=5), feature(kind=9999)]}, drainage)
    assert records == []


def test_covered_stretches_are_kept_and_flagged(drainage):
    """Rio canalizado continua sendo o caminho da água."""
    records = parse_features({"features": [feature(covered=1)]}, drainage)
    assert records[0]["covered"] is True


def test_a_blank_name_becomes_missing(drainage):
    """O cadastro grava um espaço em branco nos trechos sem nome."""
    records = parse_features({"features": [feature(name=" ")]}, drainage)
    assert records[0]["name"] is None


def test_a_multipart_path_becomes_a_multiline(drainage):
    paths = [[[0, 0], [10, 0]], [[20, 0], [30, 0]]]
    records = parse_features({"features": [feature(paths=paths)]}, drainage)
    assert records[0]["geometry"].geom_type == "MultiLineString"


def test_a_feature_without_geometry_is_skipped(drainage):
    broken = feature()
    broken["geometry"] = {}
    assert parse_features({"features": [broken]}, drainage) == []


def test_a_service_error_is_raised(drainage):
    with pytest.raises(DrainageError, match="recusou"):
        parse_features({"error": {"code": 503}}, drainage)


# --------------------------------------------------------------------------- #
# Completude do download
# --------------------------------------------------------------------------- #


def test_a_partial_download_is_refused():
    """O defeito que motivou a checagem: um arquivo com 12.249 trechos de um
    serviço que pode ter mais parece inteiro a quem só o abre."""
    with pytest.raises(DrainageError, match="incompleto"):
        check_complete(received=12_249, expected=14_251)


def test_a_complete_download_passes():
    check_complete(received=14_251, expected=14_251)


def test_an_unknown_count_does_not_block():
    check_complete(received=10, expected=None)


def test_the_summary_separates_covered_length(drainage):
    import geopandas as gpd

    records = parse_features(
        {"features": [feature(1), feature(2, covered=1, paths=[[[0, 0], [300, 0]]])]},
        drainage,
    )
    stats = summarise(gpd.GeoDataFrame(records, crs="EPSG:31982"))
    assert stats["total_km"] == pytest.approx(0.4)
    assert stats["covered_km"] == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# Distância
# --------------------------------------------------------------------------- #


def test_the_channel_itself_is_at_distance_zero():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, :] = True
    assert (distance_to_true(mask, 10.0)[2, :] == 0).all()


def test_distance_is_in_metres_not_pixels():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, :] = True
    distance = distance_to_true(mask, 10.0)
    assert distance[3, 2] == pytest.approx(30.0)


def test_distance_is_euclidean():
    """Na diagonal vale Pitágoras, não a soma dos catetos."""
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True
    assert distance_to_true(mask, 10.0)[3, 4] == pytest.approx(50.0)


def test_a_grid_without_any_channel_is_rejected():
    """Rede e grade em projeções diferentes produzem exatamente isto."""
    with pytest.raises(DrainageDistanceError, match="nenhum pixel"):
        distance_to_true(np.zeros((4, 4), dtype=bool), 10.0)


def test_a_non_positive_resolution_is_rejected():
    with pytest.raises(DrainageDistanceError, match="positiva"):
        distance_to_true(np.ones((4, 4), dtype=bool), 0.0)


# --------------------------------------------------------------------------- #
# Configuração e registro
# --------------------------------------------------------------------------- #


def test_both_stages_are_registered():
    assert STAGES_BY_NAME["acquire-drainage"].implemented
    assert STAGES_BY_NAME["build-drainage"].implemented


def test_the_paging_field_must_be_downloaded():
    data = copy.deepcopy(raw_config())
    data["drainage"]["out_fields"].remove("objectid")
    with pytest.raises(ConfigError, match="order_by_field"):
        Config.from_dict(data, root=find_repo_root())


def test_a_page_larger_than_the_service_cap_is_rejected():
    data = copy.deepcopy(raw_config())
    data["drainage"]["page_size"] = 5000
    with pytest.raises(ConfigError, match="page_size"):
        Config.from_dict(data, root=find_repo_root())


# --------------------------------------------------------------------------- #
# Talvegues derivados do relevo
# --------------------------------------------------------------------------- #


def valley(rows=30, cols=31):
    """Um vale em V que desce para o sul: o talvegue é a coluna do meio."""
    y, x = np.mgrid[0:rows, 0:cols]
    return 100.0 + np.abs(x - cols // 2) * 2.0 - y * 0.5


def test_the_valley_floor_collects_the_slopes():
    from floodrisk.features.flow import contributing_area

    area = contributing_area(valley())
    middle = area.shape[1] // 2
    assert area[-2, middle] > 0.5 * area.size
    assert area[-2, 2] < 0.05 * area.size


def test_contributing_area_grows_downstream():
    from floodrisk.features.flow import contributing_area

    area = contributing_area(valley())
    floor = area[1:-1, area.shape[1] // 2]
    assert (np.diff(floor) > 0).all()


def test_a_pit_does_not_swallow_the_flow():
    """Sem o preenchimento, o escoamento morreria no poço e o talvegue sumiria
    a jusante — que é o que ruído de DEM faz em área urbana."""
    from floodrisk.features.flow import contributing_area

    surface = valley()
    middle = surface.shape[1] // 2
    surface[10, middle] -= 50.0
    area = contributing_area(surface)
    assert area[-2, middle] > 0.5 * area.size


def test_filling_never_lowers_the_terrain():
    from floodrisk.features.flow import fill_depressions

    surface = np.random.default_rng(0).normal(900, 5, (25, 25))
    assert (fill_depressions(surface) >= surface).all()


def test_every_interior_cell_drains_after_filling():
    from floodrisk.features.flow import fill_depressions, flow_receivers

    surface = np.random.default_rng(1).normal(900, 5, (25, 25))
    receiver = flow_receivers(fill_depressions(surface))
    assert (receiver[1:-1, 1:-1] >= 0).all()


def test_a_flat_surface_is_drained_not_rejected():
    from floodrisk.features.flow import contributing_area

    area = contributing_area(np.full((12, 12), 900.0))
    assert area.min() >= 1 and area.sum() > area.size


def test_a_dem_with_gaps_is_rejected():
    from floodrisk.features.flow import FlowError, fill_depressions

    surface = valley()
    surface[3, 3] = np.nan
    with pytest.raises(FlowError, match="não finitos"):
        fill_depressions(surface)


def test_the_talweg_mask_comes_back_on_the_input_grid():
    from floodrisk.features.drainage import talweg_mask

    fine = np.kron(valley(), np.ones((3, 3)))
    mask = talweg_mask(fine, resolution_m=10.0, native_resolution_m=30.0, min_area_km2=0.2)
    assert mask.shape == fine.shape
    middle = fine.shape[1] // 2
    assert mask[-6, middle] and not mask[-6, 3]


def test_a_non_positive_talweg_threshold_is_rejected():
    data = copy.deepcopy(raw_config())
    data["drainage"]["talweg_min_area_km2"] = 0
    with pytest.raises(ConfigError, match="talweg_min_area_km2"):
        Config.from_dict(data, root=find_repo_root())

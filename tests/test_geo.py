from __future__ import annotations

import pytest
from shapely.geometry import box

from floodrisk.geo import bbox_polygon, grid_cells, patch_windows, snap_bounds


def test_patch_windows_cover_full_raster():
    width, height, patch, overlap = 1000, 700, 256, 32
    windows = list(patch_windows(width, height, patch, overlap))

    assert all(w.width == patch and w.height == patch for w in windows)
    assert all(w.col_off + w.width <= width for w in windows)
    assert all(w.row_off + w.height <= height for w in windows)
    # A borda direita e a inferior precisam estar cobertas.
    assert max(w.col_off + w.width for w in windows) == width
    assert max(w.row_off + w.height for w in windows) == height


def test_patch_windows_no_duplicates():
    windows = list(patch_windows(512, 512, 256, 0))
    assert len(windows) == len({w.as_tuple() for w in windows}) == 4


def test_patch_windows_rejects_raster_smaller_than_patch():
    with pytest.raises(ValueError, match="menor que o patch"):
        list(patch_windows(100, 100, 256, 32))


def test_patch_windows_rejects_overlap_ge_patch():
    with pytest.raises(ValueError, match="overlap"):
        list(patch_windows(1000, 1000, 256, 256))


def test_snap_bounds_expands_to_grid():
    assert snap_bounds((10.4, 20.7, 39.2, 51.1), 10.0) == (10.0, 20.0, 40.0, 60.0)


def test_snap_bounds_shrinks_when_not_expanding():
    assert snap_bounds((10.4, 20.7, 39.2, 51.1), 10.0, expand=False) == (20.0, 30.0, 30.0, 50.0)


def test_snap_bounds_rejects_zero_resolution():
    with pytest.raises(ValueError):
        snap_bounds((0, 0, 1, 1), 0)


def test_grid_cells_cover_geometry_area():
    geom = box(0, 0, 1000, 600)
    cells = grid_cells(geom, 200)
    assert len(cells) == 5 * 3
    assert sum(c.area for c in cells) == pytest.approx(geom.area)


def test_grid_cells_skip_cells_outside_geometry():
    geom = box(0, 0, 200, 200).union(box(600, 600, 800, 800))
    cells = grid_cells(geom, 200)
    assert len(cells) == 2


def test_grid_cells_clip_trims_border():
    geom = box(0, 0, 250, 250)
    clipped = grid_cells(geom, 200, clip=True)
    assert sum(c.area for c in clipped) == pytest.approx(geom.area)


def test_bbox_polygon_matches_bounds():
    bounds = [-49.396, -25.656, -49.184, -25.344]
    assert bbox_polygon(bounds).bounds == pytest.approx(tuple(bounds))

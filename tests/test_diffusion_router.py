import numpy as np
from PIL import Image

from diffusion_router_utils import (
    context_crops,
    edge_scores,
    paste_refined_centers,
    random_tiles,
    top_tiles,
    variance_scores,
)


def test_top_tiles_selects_largest_scores():
    scores = np.asarray([1.0, 4.0, 2.0, 3.0])
    assert top_tiles(scores, 2) == [1, 3]


def test_random_tiles_are_unique_and_reproducible():
    a = random_tiles(16, 4, 7)
    b = random_tiles(16, 4, 7)
    assert a == b
    assert len(set(a)) == 4


def test_edge_and_variance_scores_match_grid_size():
    x = np.zeros((32, 32, 3), dtype=float)
    x[:, 16:] = 1.0
    assert edge_scores(x, grid=4).shape == (16,)
    assert variance_scores(x, grid=4).shape == (16,)


def test_context_crop_round_trip_pastes_only_selected_center():
    base = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    crops, meta = context_crops(base, [0], grid=2, context=8)
    assert crops[0].size == (48, 48)

    refined = Image.fromarray(np.full((48, 48, 3), 255, dtype=np.uint8), mode="RGB")
    out = np.asarray(paste_refined_centers(base, [refined], meta))

    assert np.all(out[:32, :32] == 255)
    assert np.all(out[32:, :] == 0)
    assert np.all(out[:, 32:] == 0)

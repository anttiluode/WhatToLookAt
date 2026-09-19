import numpy as np

from diffusion_router_utils import correction_capture, make_tile_mask


def test_tile_mask_covers_exact_quarter_for_four_of_sixteen_tiles():
    m = np.asarray(make_tile_mask([0, 5, 10, 15], size=64, grid=4))
    assert m.shape == (64, 64)
    assert np.mean(m > 0) == 0.25


def test_capture_uses_selected_energy_fraction():
    energy = np.arange(1, 17, dtype=float)
    selected = [15, 14, 13, 12]
    got = correction_capture(energy, selected)
    expected = energy[selected].sum() / energy.sum()
    assert abs(got - expected) < 1e-12

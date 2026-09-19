from diffusion_router_utils import sparse_latent_work_fraction


def test_sparse_latent_work_fractions_match_p3_geometry():
    assert sparse_latent_work_fraction(4, 4, 0, 64) == 0.25
    assert sparse_latent_work_fraction(4, 4, 4, 64) == 0.5625
    assert sparse_latent_work_fraction(4, 4, 8, 64) == 1.0

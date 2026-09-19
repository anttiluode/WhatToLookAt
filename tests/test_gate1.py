from gate1_history_graph import (
    adjusted_rand_index,
    independently_shuffle_time,
    kmeans_farthest,
    make_motion_history,
    run_gate1,
)


def test_common_fate_order_is_learnable_but_time_shuffle_is_not():
    traces, truth = make_motion_history(5, noise=0.30)
    learned = kmeans_farthest(traces.reshape((16, -1)), 4)
    shuffled = independently_shuffle_time(traces, 99)
    shuffled_labels = kmeans_farthest(shuffled.reshape((16, -1)), 4)

    assert adjusted_rand_index(truth, learned) == 1.0
    assert adjusted_rand_index(truth, shuffled_labels) < 0.5


def test_gate1_history_graph_buys_future_measurements():
    receipt = run_gate1(n_worlds=16)
    assert receipt.median_common_fate_ari == 1.0
    assert receipt.median_time_shuffled_ari < 0.25
    assert receipt.active_learned_measurements_median == 4.0
    assert receipt.active_learned_success_fraction >= 0.90
    assert receipt.learned_common_fate.first_count_at_90pct_success is not None
    assert receipt.learned_common_fate.first_count_at_90pct_success <= 24
    assert receipt.time_shuffled_history.first_count_at_90pct_success is None
    assert receipt.coordinate_memory.first_count_at_90pct_success is None
    assert receipt.spatial_lattice.first_count_at_90pct_success is None

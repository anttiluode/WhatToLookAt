"""Gate 4: relation model mismatch buys extra sensing from a cheap residual sketch.

Gate 3 still made every expensive patch inside one relation exactly equal.
Gate 4 weakens that assumption.

Each hidden object has a shared high-resolution base plus local patch residuals.
Most residuals are tiny; a controlled number are large innovations. A cheap
8-scalar linear guide projection is observed for every patch and counted in the
sensing budget.

One high-resolution anchor patch is selected per relation using only the guide
measurements. The guide residual from that anchor is then used to decide whether
another patch can safely inherit the anchor's high-resolution content or needs
its own expensive read.

The decision threshold is learned on training worlds against an explicit local
quality criterion (would inheriting the anchor fall below 40 dB for this
patch?), checked on a separate validation set, and frozen.

Attackers:
- fixed one-per-relation: never buy extra reads;
- equal-cost random extra: buy exactly the same number of extra expensive reads
  as the adaptive policy, but at random non-anchor patches;
- oracle residual: buy exactly the patches whose true anchor-copy error crosses
  the 40 dB local threshold;
- full scan.

This asks a narrow question: can a cheap current measurement audit when a
learned relation has stopped being predictive enough to justify compression?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

N_OBJECTS = 4
PARTS_PER_OBJECT = 4
N_PARTS = N_OBJECTS * PARTS_PER_OBJECT

GUIDE_D = 8
CONTENT_H = 8
CONTENT_W = 8
CONTENT_C = 3
CONTENT_D = CONTENT_H * CONTENT_W * CONTENT_C

GUIDE_COST = N_PARTS * GUIDE_D
FULL_SCAN_COST = N_PARTS * CONTENT_D
QUALITY_DB = 40.0
LOCAL_MSE_LIMIT = 10.0 ** (-QUALITY_DB / 10.0)

INNOVATION_LEVELS = (0, 2, 4, 6, 8)
LOW_RESIDUAL_AMPLITUDE = 0.003
HIGH_RESIDUAL_AMPLITUDE = 0.18


@dataclass(frozen=True)
class PolicyLevelReceipt:
    innovation_patches: int
    mean_oracle_extra_reads: float
    success_fraction: float
    mean_expensive_patch_reads: float
    mean_total_scalar_cost: float
    mean_total_cost_fraction_of_full_scan: float
    median_psnr_db: float


@dataclass(frozen=True)
class Gate4Receipt:
    training_worlds: int
    validation_worlds: int
    test_worlds_per_level: int
    quality_threshold_psnr_db: float
    learned_guide_residual_threshold: float
    training_balanced_accuracy: float
    validation_balanced_accuracy: float
    guide_cost_scalars: int
    expensive_patch_cost_scalars: int
    full_scan_cost_scalars: int
    fixed_one_per_relation: list[PolicyLevelReceipt]
    residual_adaptive: list[PolicyLevelReceipt]
    equal_cost_random_extra: list[PolicyLevelReceipt]
    oracle_residual: list[PolicyLevelReceipt]
    full_scan: list[PolicyLevelReceipt]


def make_base_content_and_projection(
    seed: int = 111,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.15, 0.85, size=(N_OBJECTS, CONTENT_D))
    projection = rng.normal(size=(GUIDE_D, CONTENT_D)) / np.sqrt(CONTENT_D)
    return base, projection


BASE_CONTENT, GUIDE_PROJECTION = make_base_content_and_projection()


def make_world(
    seed: int,
    innovation_patches: int,
    low_amplitude: float = LOW_RESIDUAL_AMPLITUDE,
    high_amplitude: float = HIGH_RESIDUAL_AMPLITUDE,
) -> dict:
    rng = np.random.default_rng(seed)
    groups = np.repeat(
        np.arange(N_OBJECTS, dtype=np.int64),
        PARTS_PER_OBJECT,
    )
    innovations = np.zeros(N_PARTS, dtype=bool)
    if innovation_patches:
        innovations[
            rng.choice(N_PARTS, int(innovation_patches), replace=False)
        ] = True

    content = np.empty((N_PARTS, CONTENT_D), dtype=np.float64)
    for patch, group in enumerate(groups):
        amplitude = high_amplitude if innovations[patch] else low_amplitude
        direction = rng.normal(size=CONTENT_D)
        direction /= np.sqrt(np.mean(direction**2))
        content[patch] = np.clip(
            BASE_CONTENT[int(group)] + amplitude * direction,
            0.0,
            1.0,
        )

    guide = content @ GUIDE_PROJECTION.T
    return {
        "groups": groups,
        "content": content,
        "guide": guide,
        "innovations": innovations,
    }


def relation_audit(world: dict) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Return guide residual score, oracle read-needed label, and anchors."""
    groups = np.asarray(world["groups"])
    content = np.asarray(world["content"])
    guide = np.asarray(world["guide"])

    score = np.zeros(N_PARTS, dtype=np.float64)
    needs_read = np.zeros(N_PARTS, dtype=bool)
    anchors: dict[int, int] = {}

    for group in range(N_OBJECTS):
        members = np.flatnonzero(groups == group)
        centroid = guide[members].mean(axis=0)
        centroid_distance = np.linalg.norm(
            guide[members] - centroid[None, :],
            axis=1,
        )
        anchor = int(members[np.argmin(centroid_distance)])
        anchors[group] = anchor

        score[members] = np.linalg.norm(
            guide[members] - guide[anchor][None, :],
            axis=1,
        )
        for patch in members:
            mse = float(np.mean((content[patch] - content[anchor]) ** 2))
            needs_read[patch] = mse > LOCAL_MSE_LIMIT

    return score, needs_read, anchors


def balanced_accuracy(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> float:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    positive = (
        float(np.mean(predicted[truth])) if np.any(truth) else 1.0
    )
    negative = (
        float(np.mean(~predicted[~truth])) if np.any(~truth) else 1.0
    )
    return 0.5 * (positive + negative)


def learn_threshold(worlds: list[dict]) -> tuple[float, float]:
    scores = []
    labels = []
    for world in worlds:
        score, need, _ = relation_audit(world)
        scores.append(score)
        labels.append(need)

    values = np.concatenate(scores)
    truth = np.concatenate(labels)
    candidates = np.linspace(float(values.min()), float(values.max()), 201)

    best_threshold = float(candidates[0])
    best_score = -1.0
    for threshold in candidates:
        score = balanced_accuracy(truth, values >= threshold)
        if score > best_score + 1e-12:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score


def threshold_accuracy(worlds: list[dict], threshold: float) -> float:
    scores = []
    labels = []
    for world in worlds:
        score, need, _ = relation_audit(world)
        scores.append(score)
        labels.append(need)
    values = np.concatenate(scores)
    truth = np.concatenate(labels)
    return balanced_accuracy(truth, values >= threshold)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(reference) - np.asarray(estimate)) ** 2))
    if mse <= 1e-14:
        return 120.0
    return float(10.0 * np.log10(1.0 / mse))


def evaluate_world(
    world: dict,
    policy: str,
    threshold: float,
    random_seed: int,
) -> tuple[float, int, int, int]:
    score, oracle_need, anchors = relation_audit(world)
    groups = np.asarray(world["groups"])
    content = np.asarray(world["content"])

    estimate = np.empty_like(content)
    reads: set[int] = set()

    for group in range(N_OBJECTS):
        members = np.flatnonzero(groups == group)
        anchor = int(anchors[group])
        reads.add(anchor)
        estimate[members] = content[anchor]

    flags = np.zeros(N_PARTS, dtype=bool)
    guide_cost = GUIDE_COST

    if policy == "fixed":
        pass
    elif policy == "adaptive":
        flags = score >= threshold
    elif policy == "equal_cost_random":
        adaptive_flags = score >= threshold
        extra = [
            patch
            for patch in range(N_PARTS)
            if adaptive_flags[patch] and patch not in reads
        ]
        candidates = [
            patch for patch in range(N_PARTS) if patch not in reads
        ]
        rng = np.random.default_rng(random_seed)
        if extra:
            chosen = rng.choice(
                candidates,
                size=len(extra),
                replace=False,
            )
            flags[np.asarray(chosen, dtype=np.int64)] = True
    elif policy == "oracle":
        flags = oracle_need.copy()
    elif policy == "full":
        flags[:] = True
        guide_cost = 0
    else:
        raise ValueError(policy)

    for patch in np.flatnonzero(flags):
        reads.add(int(patch))
        estimate[patch] = content[patch]

    total_cost = guide_cost + len(reads) * CONTENT_D
    return (
        psnr(content, estimate),
        len(reads),
        total_cost,
        int(np.sum(oracle_need)),
    )


def evaluate_level(
    worlds: list[dict],
    policy: str,
    threshold: float,
) -> PolicyLevelReceipt:
    results = [
        evaluate_world(
            world,
            policy,
            threshold,
            random_seed=50000
            + index
            + 1000 * int(np.sum(np.asarray(world["innovations"], dtype=bool))),
        )
        for index, world in enumerate(worlds)
    ]

    mean_cost = float(np.mean([item[2] for item in results]))
    return PolicyLevelReceipt(
        innovation_patches=int(
            np.sum(np.asarray(worlds[0]["innovations"], dtype=bool))
        ),
        mean_oracle_extra_reads=float(
            np.mean([item[3] for item in results])
        ),
        success_fraction=float(
            np.mean([item[0] >= QUALITY_DB for item in results])
        ),
        mean_expensive_patch_reads=float(
            np.mean([item[1] for item in results])
        ),
        mean_total_scalar_cost=mean_cost,
        mean_total_cost_fraction_of_full_scan=mean_cost / FULL_SCAN_COST,
        median_psnr_db=float(
            np.median([item[0] for item in results])
        ),
    )


def run_gate4(
    training_worlds: int = 100,
    validation_worlds: int = 50,
    test_worlds_per_level: int = 80,
) -> Gate4Receipt:
    train = [
        make_world(
            1000 + index,
            INNOVATION_LEVELS[index % len(INNOVATION_LEVELS)],
        )
        for index in range(int(training_worlds))
    ]
    validation = [
        make_world(
            3000 + index,
            INNOVATION_LEVELS[index % len(INNOVATION_LEVELS)],
        )
        for index in range(int(validation_worlds))
    ]

    threshold, training_score = learn_threshold(train)
    validation_score = threshold_accuracy(validation, threshold)

    levels = {
        int(level): [
            make_world(10000 + 1000 * int(level) + index, int(level))
            for index in range(int(test_worlds_per_level))
        ]
        for level in INNOVATION_LEVELS
    }

    def collect(policy: str) -> list[PolicyLevelReceipt]:
        return [
            evaluate_level(levels[int(level)], policy, threshold)
            for level in INNOVATION_LEVELS
        ]

    return Gate4Receipt(
        training_worlds=int(training_worlds),
        validation_worlds=int(validation_worlds),
        test_worlds_per_level=int(test_worlds_per_level),
        quality_threshold_psnr_db=QUALITY_DB,
        learned_guide_residual_threshold=float(threshold),
        training_balanced_accuracy=float(training_score),
        validation_balanced_accuracy=float(validation_score),
        guide_cost_scalars=GUIDE_COST,
        expensive_patch_cost_scalars=CONTENT_D,
        full_scan_cost_scalars=FULL_SCAN_COST,
        fixed_one_per_relation=collect("fixed"),
        residual_adaptive=collect("adaptive"),
        equal_cost_random_extra=collect("equal_cost_random"),
        oracle_residual=collect("oracle"),
        full_scan=collect("full"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-worlds", type=int, default=100)
    parser.add_argument("--validation-worlds", type=int, default=50)
    parser.add_argument("--test-worlds", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate4_summary.json"),
    )
    args = parser.parse_args()
    receipt = run_gate4(
        training_worlds=args.train_worlds,
        validation_worlds=args.validation_worlds,
        test_worlds_per_level=args.test_worlds,
    )
    payload = asdict(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

"""Gate 2: local relation confidence controls physical sensing budget.

Gate 1 assumed patch correspondence was trustworthy. Gate 2 attacks that
scaffold. A synthetic tracker proposes current-patch -> historical-patch
matches. Some proposals are deliberately swapped across objects. The tracker
also emits a local confidence value. A threshold is learned on separate
training worlds, then frozen.

Policies on held-out worlds:
- always trust: cheapest, but stale/wrong correspondence retains full authority;
- globally cautious: measure every patch, robust but expensive;
- confidence adaptive: keep relation sharing only for trusted matches and make
  low-confidence patches pay for their own measurements;
- shuffled confidence: same sensing budget distribution, wrong locations;
- oracle uncertainty: ceiling using true correspondence correctness.

The intended receipt is not that confidence is magical -- it is synthetic in
this gate. The claim is that *local* uncertainty can be converted into a local
measurement budget instead of forcing either global trust or global caution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gate1_history_graph import N_OBJECTS, N_PARTS, PARTS_PER_OBJECT, expand_patch_values
from what_to_look_at import COLORS, active_component_mask, group_harmonic_reconstruct, psnr

QUALITY_DB = 40.0


@dataclass(frozen=True)
class PolicyLevelReceipt:
    wrong_matches: int
    success_fraction: float
    mean_measurements: float
    median_psnr_db: float


@dataclass(frozen=True)
class Gate2Receipt:
    training_worlds: int
    test_worlds_per_level: int
    learned_confidence_threshold: float
    training_balanced_accuracy: float
    always_trust: list[PolicyLevelReceipt]
    confidence_adaptive: list[PolicyLevelReceipt]
    global_cautious: list[PolicyLevelReceipt]
    shuffled_confidence: list[PolicyLevelReceipt]
    oracle_uncertainty: list[PolicyLevelReceipt]


def make_correspondence_world(
    seed: int,
    swap_pairs: int,
    correct_mean: float = 0.90,
    wrong_mean: float = 0.25,
    confidence_sd: float = 0.12,
) -> dict:
    """Create a future patch arrangement plus noisy tracker proposals/confidence."""
    rng = np.random.default_rng(seed)
    patch_ids = np.arange(N_PARTS, dtype=np.int64)
    rng.shuffle(patch_ids)
    true_groups = patch_ids // PARTS_PER_OBJECT
    proposed_ids = patch_ids.copy()

    available = list(range(N_PARTS))
    wrong = np.zeros(N_PARTS, dtype=bool)
    for _ in range(int(swap_pairs)):
        candidates = [
            (i, j)
            for pos, i in enumerate(available)
            for j in available[pos + 1 :]
            if true_groups[i] != true_groups[j]
        ]
        if not candidates:
            break
        i, j = candidates[int(rng.integers(len(candidates)))]
        proposed_ids[i], proposed_ids[j] = proposed_ids[j], proposed_ids[i]
        wrong[i] = True
        wrong[j] = True
        available.remove(i)
        available.remove(j)

    confidence = np.where(
        wrong,
        rng.normal(wrong_mean, confidence_sd, N_PARTS),
        rng.normal(correct_mean, confidence_sd, N_PARTS),
    )
    confidence = np.clip(confidence, 0.0, 1.0)
    return {
        "patch_ids": patch_ids,
        "true_groups": true_groups,
        "proposed_ids": proposed_ids,
        "correct": ~wrong,
        "confidence": confidence,
    }


def balanced_accuracy(correct: np.ndarray, predicted_correct: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=bool)
    pred = np.asarray(predicted_correct, dtype=bool)
    tpr = float(np.mean(pred[correct])) if np.any(correct) else 1.0
    tnr = float(np.mean(~pred[~correct])) if np.any(~correct) else 1.0
    return 0.5 * (tpr + tnr)


def learn_confidence_threshold(training_worlds: list[dict]) -> tuple[float, float]:
    truth = np.concatenate([w["correct"] for w in training_worlds])
    conf = np.concatenate([w["confidence"] for w in training_worlds])
    best_score = -1.0
    best_threshold = 0.5
    for threshold in np.linspace(0.10, 0.90, 33):
        score = balanced_accuracy(truth, conf >= threshold)
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def patch_scene(true_groups: np.ndarray) -> np.ndarray:
    blocks = COLORS[np.asarray(true_groups)].reshape((N_OBJECTS, N_OBJECTS, 3))
    return np.repeat(np.repeat(blocks, 4, axis=0), 4, axis=1)


def expand_groups(patch_groups: np.ndarray) -> np.ndarray:
    return expand_patch_values(np.asarray(patch_groups).reshape((N_OBJECTS, N_OBJECTS)))


def active_score(scene: np.ndarray, pixel_groups: np.ndarray) -> tuple[float, int]:
    mask = active_component_mask(pixel_groups)
    observed = np.zeros_like(scene)
    observed[mask] = scene[mask]
    recovered = group_harmonic_reconstruct(mask, observed, pixel_groups)
    return psnr(scene, recovered), int(mask.sum())


def trusted_groups(world: dict) -> np.ndarray:
    return np.asarray(world["proposed_ids"]) // PARTS_PER_OBJECT


def confidence_groups(world: dict, threshold: float, confidence: np.ndarray | None = None) -> np.ndarray:
    groups = trusted_groups(world).copy().astype(np.int64)
    conf = np.asarray(world["confidence"] if confidence is None else confidence)
    next_label = N_OBJECTS
    for idx in range(N_PARTS):
        if conf[idx] < threshold:
            groups[idx] = next_label
            next_label += 1
    return groups


def oracle_uncertainty_groups(world: dict) -> np.ndarray:
    groups = trusted_groups(world).copy().astype(np.int64)
    next_label = N_OBJECTS
    for idx in range(N_PARTS):
        if not bool(world["correct"][idx]):
            groups[idx] = next_label
            next_label += 1
    return groups


def evaluate_policy_level(worlds: list[dict], policy: str, threshold: float) -> PolicyLevelReceipt:
    scores = []
    budgets = []
    for world_idx, world in enumerate(worlds):
        scene = patch_scene(world["true_groups"])
        if policy == "always_trust":
            patch_groups = trusted_groups(world)
        elif policy == "global_cautious":
            patch_groups = np.arange(N_PARTS, dtype=np.int64)
        elif policy == "confidence_adaptive":
            patch_groups = confidence_groups(world, threshold)
        elif policy == "shuffled_confidence":
            conf = np.asarray(world["confidence"]).copy()
            rng = np.random.default_rng(50000 + world_idx + 1000 * int(np.sum(~world["correct"])))
            rng.shuffle(conf)
            patch_groups = confidence_groups(world, threshold, conf)
        elif policy == "oracle_uncertainty":
            patch_groups = oracle_uncertainty_groups(world)
        else:
            raise ValueError(policy)

        score, budget = active_score(scene, expand_groups(patch_groups))
        scores.append(score)
        budgets.append(budget)

    return PolicyLevelReceipt(
        wrong_matches=int(np.sum(~worlds[0]["correct"])),
        success_fraction=float(np.mean(np.asarray(scores) >= QUALITY_DB)),
        mean_measurements=float(np.mean(budgets)),
        median_psnr_db=float(np.median(scores)),
    )


def run_gate2(training_worlds: int = 80, test_worlds_per_level: int = 80) -> Gate2Receipt:
    train = []
    for idx in range(int(training_worlds)):
        swap_pairs = idx % 4
        train.append(make_correspondence_world(1000 + idx, swap_pairs))
    threshold, train_ba = learn_confidence_threshold(train)

    levels: dict[int, list[dict]] = {}
    for swap_pairs in range(4):
        levels[swap_pairs] = [
            make_correspondence_world(10000 + 1000 * swap_pairs + idx, swap_pairs)
            for idx in range(int(test_worlds_per_level))
        ]

    def collect(policy: str) -> list[PolicyLevelReceipt]:
        return [evaluate_policy_level(levels[level], policy, threshold) for level in range(4)]

    return Gate2Receipt(
        training_worlds=int(training_worlds),
        test_worlds_per_level=int(test_worlds_per_level),
        learned_confidence_threshold=float(threshold),
        training_balanced_accuracy=float(train_ba),
        always_trust=collect("always_trust"),
        confidence_adaptive=collect("confidence_adaptive"),
        global_cautious=collect("global_cautious"),
        shuffled_confidence=collect("shuffled_confidence"),
        oracle_uncertainty=collect("oracle_uncertainty"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-worlds", type=int, default=80)
    parser.add_argument("--test-worlds", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("results/gate2_summary.json"))
    args = parser.parse_args()
    receipt = run_gate2(args.train_worlds, args.test_worlds)
    payload = asdict(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

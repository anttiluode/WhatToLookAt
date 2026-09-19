"""Gate 3: image-derived confidence controls the sensing budget.

Gate 2 proved the budgeting mechanism with a synthetic confidence scalar.
Gate 3 removes that oracle. Each tracked patch has:

- a cheap 2x4 grayscale guide image (8 scalar samples);
- an expensive 8x8 RGB content patch (192 scalar samples).

The guide image is used only for correspondence. The high-resolution content is
what must be reconstructed. Ambiguous guide patches are blended toward a
cross-object look-alike, so nearest-template matching can choose the wrong
historical identity.

Three confidence diagnostics are calibrated on training worlds:
- best-vs-second-best template margin;
- negative best-match error;
- forward/backward cycle consistency.

The cue is selected on a separate validation set and frozen. On untouched test
worlds, low-confidence matches are removed from the relation-sharing component
and pay for their own expensive content measurement.

All sensing cost is explicit:
    guide cost + number_of_expensive_patch_reads * 192.

A full high-resolution scan costs 16 * 192 = 3072 scalar samples and does not
need the guide channel.
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

GUIDE_H = 2
GUIDE_W = 4
GUIDE_D = GUIDE_H * GUIDE_W
CONTENT_H = 8
CONTENT_W = 8
CONTENT_C = 3
CONTENT_D = CONTENT_H * CONTENT_W * CONTENT_C

FULL_SCAN_COST = N_PARTS * CONTENT_D
GUIDE_COST = N_PARTS * GUIDE_D
QUALITY_DB = 40.0

AMBIGUITY_LEVELS = (0, 2, 4, 6)


@dataclass(frozen=True)
class CueReceipt:
    name: str
    threshold: float
    training_balanced_accuracy: float
    validation_balanced_accuracy: float


@dataclass(frozen=True)
class PolicyLevelReceipt:
    ambiguous_guides: int
    mean_wrong_matches: float
    success_fraction: float
    mean_expensive_patch_reads: float
    mean_total_scalar_cost: float
    mean_total_cost_fraction_of_full_scan: float
    median_psnr_db: float


@dataclass(frozen=True)
class Gate3Receipt:
    training_worlds: int
    validation_worlds: int
    test_worlds_per_level: int
    selected_cue: str
    selected_threshold: float
    guide_cost_scalars: int
    expensive_patch_cost_scalars: int
    full_scan_cost_scalars: int
    candidate_cues: list[CueReceipt]
    always_trust: list[PolicyLevelReceipt]
    image_confidence_adaptive: list[PolicyLevelReceipt]
    shuffled_confidence: list[PolicyLevelReceipt]
    oracle_uncertainty: list[PolicyLevelReceipt]
    full_scan: list[PolicyLevelReceipt]


def make_guide_templates(seed: int = 123) -> np.ndarray:
    """Fixed low-resolution appearance templates for the 16 patch identities."""
    rng = np.random.default_rng(seed)
    templates = rng.normal(size=(N_PARTS, GUIDE_D))
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    return templates


def make_object_contents(seed: int = 456) -> np.ndarray:
    """High-resolution RGB content shared by all four patches of each object."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.95, size=(N_OBJECTS, CONTENT_D))


GUIDE_TEMPLATES = make_guide_templates()
OBJECT_CONTENT = make_object_contents()


def make_world(
    seed: int,
    ambiguous_guides: int,
    guide_noise: float = 0.06,
) -> dict:
    """Make one tracked world with a cheap guide channel and expensive content.

    Current positions contain a random permutation of the 16 historical patch
    identities. Clean guides are noisy versions of the correct historical
    thumbnail. Ambiguous guides are blended slightly past the midpoint toward
    the nearest cross-object look-alike, which makes the matching problem real
    rather than directly writing a wrong identity into the tracker.
    """
    rng = np.random.default_rng(seed)
    true_ids = np.arange(N_PARTS, dtype=np.int64)
    rng.shuffle(true_ids)

    guide = GUIDE_TEMPLATES[true_ids] + rng.normal(
        scale=guide_noise, size=(N_PARTS, GUIDE_D)
    )

    attacked = np.zeros(N_PARTS, dtype=bool)
    if ambiguous_guides:
        positions = rng.choice(N_PARTS, int(ambiguous_guides), replace=False)
        for pos in positions:
            true_id = int(true_ids[pos])
            candidates = [
                other
                for other in range(N_PARTS)
                if other // PARTS_PER_OBJECT != true_id // PARTS_PER_OBJECT
            ]
            distractor = min(
                candidates,
                key=lambda other: float(
                    np.linalg.norm(
                        GUIDE_TEMPLATES[true_id] - GUIDE_TEMPLATES[other]
                    )
                ),
            )
            alpha = float(rng.uniform(0.52, 0.62))
            guide[pos] = (
                (1.0 - alpha) * GUIDE_TEMPLATES[true_id]
                + alpha * GUIDE_TEMPLATES[distractor]
                + rng.normal(scale=guide_noise, size=GUIDE_D)
            )
            attacked[pos] = True

    distances = np.linalg.norm(
        guide[:, None, :] - GUIDE_TEMPLATES[None, :, :], axis=-1
    )
    proposed_ids = np.argmin(distances, axis=1)
    ordered = np.sort(distances, axis=1)
    best_error = ordered[:, 0]
    second_error = ordered[:, 1]

    # Mutual nearest-neighbour / forward-backward match consistency.
    template_to_current = np.argmin(distances, axis=0)
    cycle = np.asarray(
        [
            float(template_to_current[int(proposed_ids[pos])] == pos)
            for pos in range(N_PARTS)
        ],
        dtype=np.float64,
    )

    return {
        "true_ids": true_ids,
        "proposed_ids": proposed_ids,
        "correct": proposed_ids == true_ids,
        "attacked": attacked,
        "margin": second_error - best_error,
        "neg_error": -best_error,
        "cycle": cycle,
    }


def balanced_accuracy(correct: np.ndarray, predicted_correct: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=bool)
    predicted = np.asarray(predicted_correct, dtype=bool)
    positive = (
        float(np.mean(predicted[correct])) if np.any(correct) else 1.0
    )
    negative = (
        float(np.mean(~predicted[~correct])) if np.any(~correct) else 1.0
    )
    return 0.5 * (positive + negative)


def learn_threshold(worlds: list[dict], cue: str) -> tuple[float, float]:
    values = np.concatenate([np.asarray(world[cue]) for world in worlds])
    truth = np.concatenate([np.asarray(world["correct"]) for world in worlds])

    thresholds = np.linspace(float(values.min()), float(values.max()), 101)
    best_threshold = float(thresholds[0])
    best_score = -1.0
    for threshold in thresholds:
        score = balanced_accuracy(truth, values >= threshold)
        if score > best_score + 1e-12:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def score_threshold(worlds: list[dict], cue: str, threshold: float) -> float:
    values = np.concatenate([np.asarray(world[cue]) for world in worlds])
    truth = np.concatenate([np.asarray(world["correct"]) for world in worlds])
    return balanced_accuracy(truth, values >= threshold)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(reference) - np.asarray(estimate)) ** 2))
    if mse <= 1e-14:
        return 120.0
    return float(10.0 * np.log10(1.0 / mse))


def proposed_relation_groups(world: dict) -> np.ndarray:
    return np.asarray(world["proposed_ids"], dtype=np.int64) // PARTS_PER_OBJECT


def confidence_groups(
    world: dict,
    cue: str,
    threshold: float,
    values: np.ndarray | None = None,
) -> np.ndarray:
    groups = proposed_relation_groups(world).copy()
    confidence = np.asarray(world[cue] if values is None else values)
    next_group = N_OBJECTS
    for pos in range(N_PARTS):
        if confidence[pos] < threshold:
            groups[pos] = next_group
            next_group += 1
    return groups


def oracle_uncertainty_groups(world: dict) -> np.ndarray:
    groups = proposed_relation_groups(world).copy()
    next_group = N_OBJECTS
    for pos in range(N_PARTS):
        if not bool(world["correct"][pos]):
            groups[pos] = next_group
            next_group += 1
    return groups


def reconstruct_from_groups(
    world: dict,
    groups: np.ndarray,
) -> tuple[float, int]:
    true_object = np.asarray(world["true_ids"], dtype=np.int64) // PARTS_PER_OBJECT
    reference = OBJECT_CONTENT[true_object]
    estimate = np.zeros_like(reference)

    reads = 0
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        representative = int(members[0])
        measured_content = reference[representative]
        estimate[members] = measured_content
        reads += 1

    return psnr(reference, estimate), reads


def evaluate_level(
    worlds: list[dict],
    policy: str,
    cue: str,
    threshold: float,
) -> PolicyLevelReceipt:
    scores: list[float] = []
    reads: list[int] = []
    costs: list[float] = []
    wrong: list[int] = []

    for world_index, world in enumerate(worlds):
        if policy == "always_trust":
            groups = proposed_relation_groups(world)
            guide_cost = GUIDE_COST
        elif policy == "adaptive":
            groups = confidence_groups(world, cue, threshold)
            guide_cost = GUIDE_COST
        elif policy == "shuffled_confidence":
            values = np.asarray(world[cue]).copy()
            rng = np.random.default_rng(
                50000
                + world_index
                + 1000 * int(np.sum(np.asarray(world["attacked"], dtype=bool)))
            )
            rng.shuffle(values)
            groups = confidence_groups(world, cue, threshold, values=values)
            guide_cost = GUIDE_COST
        elif policy == "oracle":
            groups = oracle_uncertainty_groups(world)
            guide_cost = GUIDE_COST
        elif policy == "full_scan":
            groups = np.arange(N_PARTS, dtype=np.int64)
            guide_cost = 0
        else:
            raise ValueError(policy)

        score, patch_reads = reconstruct_from_groups(world, groups)
        total_cost = guide_cost + patch_reads * CONTENT_D

        scores.append(score)
        reads.append(patch_reads)
        costs.append(total_cost)
        wrong.append(int(np.sum(~np.asarray(world["correct"], dtype=bool))))

    ambiguity = int(np.sum(np.asarray(worlds[0]["attacked"], dtype=bool)))
    mean_cost = float(np.mean(costs))
    return PolicyLevelReceipt(
        ambiguous_guides=ambiguity,
        mean_wrong_matches=float(np.mean(wrong)),
        success_fraction=float(np.mean(np.asarray(scores) >= QUALITY_DB)),
        mean_expensive_patch_reads=float(np.mean(reads)),
        mean_total_scalar_cost=mean_cost,
        mean_total_cost_fraction_of_full_scan=mean_cost / FULL_SCAN_COST,
        median_psnr_db=float(np.median(scores)),
    )


def run_gate3(
    training_worlds: int = 80,
    validation_worlds: int = 40,
    test_worlds_per_level: int = 80,
) -> Gate3Receipt:
    train = [
        make_world(1000 + index, AMBIGUITY_LEVELS[index % len(AMBIGUITY_LEVELS)])
        for index in range(int(training_worlds))
    ]
    validation = [
        make_world(
            3000 + index,
            AMBIGUITY_LEVELS[index % len(AMBIGUITY_LEVELS)],
        )
        for index in range(int(validation_worlds))
    ]

    cue_receipts: list[CueReceipt] = []
    for cue in ("margin", "neg_error", "cycle"):
        threshold, training_score = learn_threshold(train, cue)
        validation_score = score_threshold(validation, cue, threshold)
        cue_receipts.append(
            CueReceipt(
                name=cue,
                threshold=float(threshold),
                training_balanced_accuracy=float(training_score),
                validation_balanced_accuracy=float(validation_score),
            )
        )

    selected = max(
        cue_receipts,
        key=lambda item: (item.validation_balanced_accuracy, item.training_balanced_accuracy),
    )

    levels: dict[int, list[dict]] = {}
    for ambiguity in AMBIGUITY_LEVELS:
        levels[int(ambiguity)] = [
            make_world(10000 + 1000 * int(ambiguity) + index, int(ambiguity))
            for index in range(int(test_worlds_per_level))
        ]

    def collect(policy: str) -> list[PolicyLevelReceipt]:
        return [
            evaluate_level(
                levels[int(ambiguity)],
                policy,
                selected.name,
                selected.threshold,
            )
            for ambiguity in AMBIGUITY_LEVELS
        ]

    return Gate3Receipt(
        training_worlds=int(training_worlds),
        validation_worlds=int(validation_worlds),
        test_worlds_per_level=int(test_worlds_per_level),
        selected_cue=selected.name,
        selected_threshold=float(selected.threshold),
        guide_cost_scalars=GUIDE_COST,
        expensive_patch_cost_scalars=CONTENT_D,
        full_scan_cost_scalars=FULL_SCAN_COST,
        candidate_cues=cue_receipts,
        always_trust=collect("always_trust"),
        image_confidence_adaptive=collect("adaptive"),
        shuffled_confidence=collect("shuffled_confidence"),
        oracle_uncertainty=collect("oracle"),
        full_scan=collect("full_scan"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-worlds", type=int, default=80)
    parser.add_argument("--validation-worlds", type=int, default=40)
    parser.add_argument("--test-worlds", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate3_summary.json"),
    )
    args = parser.parse_args()
    receipt = run_gate3(
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

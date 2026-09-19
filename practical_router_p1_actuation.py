#!/usr/bin/env python3
"""Practical router P1: separate *where to spend compute* from *what local action does*.

P0 found something stronger than a simple miss: every crop-refinement route
moved farther from the full-frame teacher, and all routes were slower.

This CPU-only diagnostic consumes the committed P0 images and asks two separate
questions:

1. Selection: did a selector actually choose tiles containing a large fraction
   of the teacher correction energy?
2. Actuation: when the independent crop refiner changed those selected tiles,
   did that change point in the same pixel-space direction as the teacher
   correction, or did it overshoot / move elsewhere?

For each selector/case we measure:
- selected correction-energy capture
- cosine(delta_crop, delta_teacher)
- least-squares blend alpha* for base + alpha * (routed-base)
- best achievable recovery with alpha clipped to [0,1]

Then we fit one global alpha per selector using seeds 100/101 only and freeze it
for seed 102. This tests whether P0 merely used too much local intervention, or
whether the local crop operator is fundamentally misaligned.

Interpretation:
- selection good + calibrated blend positive => useful cue, over-strong actuator
- selection good + blend still non-positive => cue may be fine, crop diffusion
  is the wrong local action
- selection poor => the P0 failure is already upstream in routing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

GRID = 4
SELECTORS = (
    "edge_energy",
    "local_variance",
    "gaussian_residue",
    "sihti_residue",
    "random",
)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def case_stem(prompt_index: int, seed: int) -> str:
    return f"p{prompt_index:02d}_s{int(seed)}"


def prompt_order(cases: list[dict]) -> dict[str, int]:
    order: list[str] = []
    for case in cases:
        if case["prompt"] not in order:
            order.append(case["prompt"])
    return {prompt: i + 1 for i, prompt in enumerate(order)}


def tile_energy_map(base: np.ndarray, teacher: np.ndarray, grid: int = GRID) -> np.ndarray:
    err = np.mean((teacher - base) ** 2, axis=2)
    h, w = err.shape
    if h % grid or w % grid:
        raise ValueError("image shape not divisible by grid")
    th, tw = h // grid, w // grid
    return err.reshape(grid, th, grid, tw).mean(axis=(1, 3)).ravel()


def tile_mask(shape: tuple[int, int, int], selected: list[int], grid: int = GRID) -> np.ndarray:
    h, w, _ = shape
    th, tw = h // grid, w // grid
    mask = np.zeros((h, w, 1), dtype=np.float64)
    for idx in selected:
        row, col = divmod(int(idx), grid)
        mask[row*th:(row+1)*th, col*tw:(col+1)*tw, 0] = 1.0
    return mask


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    den = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if den <= 1e-20:
        return 0.0
    return float(np.dot(av, bv) / den)


def recovery_for_alpha(
    base: np.ndarray,
    teacher: np.ndarray,
    delta: np.ndarray,
    alpha: float,
) -> float:
    base_err = float(np.mean((base - teacher) ** 2))
    if base_err <= 1e-20:
        return 0.0
    out = base + float(alpha) * delta
    err = float(np.mean((out - teacher) ** 2))
    return float(1.0 - err / base_err)


def evaluate_case(
    root: Path,
    pidx: int,
    case: dict,
) -> dict:
    stem = case_stem(pidx, int(case["seed"]))
    base = load_rgb(root / f"{stem}_base.png")
    teacher = load_rgb(root / f"{stem}_teacher.png")
    teacher_delta = teacher - base
    energy = tile_energy_map(base, teacher)

    methods = {}
    for method in case["methods"]:
        name = method["selector"]
        routed = load_rgb(root / f"{stem}_{name}.png")
        delta = routed - base
        selected = [int(x) for x in method["tiles"]]
        mask = tile_mask(base.shape, selected)

        selected_energy = float(energy[selected].sum())
        total_energy = float(energy.sum())
        capture = 0.25 if total_energy <= 1e-20 else selected_energy / total_energy

        d = delta * mask
        t = teacher_delta * mask
        cosine = safe_cosine(d, t)

        dd = float(np.sum(d * d))
        dt = float(np.sum(d * t))
        alpha_unclipped = 0.0 if dd <= 1e-20 else dt / dd
        alpha_clipped = float(np.clip(alpha_unclipped, 0.0, 1.0))
        oracle_blend_recovery = recovery_for_alpha(
            base, teacher, delta, alpha_clipped
        )

        methods[name] = {
            "selected_tiles": selected,
            "correction_energy_capture": float(capture),
            "actuation_cosine": float(cosine),
            "alpha_unclipped": float(alpha_unclipped),
            "alpha_clipped": alpha_clipped,
            "oracle_blend_recovery": float(oracle_blend_recovery),
            "p0_recovery": float(method["correction_recovery"]),
            "delta_dot_teacher": dt,
            "delta_energy": dd,
        }

    return {
        "prompt": case["prompt"],
        "seed": int(case["seed"]),
        "prompt_index": int(pidx),
        "methods": methods,
    }


def fit_global_alpha(cases: list[dict], selector: str) -> float:
    num = 0.0
    den = 0.0
    for case in cases:
        row = case["methods"][selector]
        num += float(row["delta_dot_teacher"])
        den += float(row["delta_energy"])
    if den <= 1e-20:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def attach_fixed_alpha_recovery(
    root: Path,
    evaluated: list[dict],
    source_cases: list[dict],
    alphas: dict[str, float],
) -> None:
    lookup = {
        (case["prompt"], int(case["seed"])): case
        for case in source_cases
    }
    pmap = prompt_order(source_cases)

    for row in evaluated:
        source = lookup[(row["prompt"], int(row["seed"]))]
        stem = case_stem(pmap[row["prompt"]], int(row["seed"]))
        base = load_rgb(root / f"{stem}_base.png")
        teacher = load_rgb(root / f"{stem}_teacher.png")
        for method in source["methods"]:
            selector = method["selector"]
            routed = load_rgb(root / f"{stem}_{selector}.png")
            delta = routed - base
            row["methods"][selector]["fixed_alpha"] = float(alphas[selector])
            row["methods"][selector]["fixed_alpha_recovery"] = recovery_for_alpha(
                base, teacher, delta, alphas[selector]
            )


def aggregate(cases: list[dict], selector: str) -> dict:
    rows = [c["methods"][selector] for c in cases]
    return {
        "cases": len(rows),
        "mean_capture": float(np.mean([r["correction_energy_capture"] for r in rows])),
        "mean_actuation_cosine": float(np.mean([r["actuation_cosine"] for r in rows])),
        "median_actuation_cosine": float(np.median([r["actuation_cosine"] for r in rows])),
        "mean_oracle_blend_recovery": float(np.mean([r["oracle_blend_recovery"] for r in rows])),
        "mean_p0_recovery": float(np.mean([r["p0_recovery"] for r in rows])),
        "mean_fixed_alpha_recovery": float(np.mean([
            r.get("fixed_alpha_recovery", 0.0) for r in rows
        ])),
        "positive_fixed_alpha_cases": int(np.sum([
            r.get("fixed_alpha_recovery", 0.0) > 0.0 for r in rows
        ])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--p0-summary",
        type=Path,
        default=Path("results/practical_diffusion_router/summary.json"),
    )
    ap.add_argument(
        "--image-root",
        type=Path,
        default=Path("results/practical_diffusion_router"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/practical_diffusion_router_p1_summary.json"),
    )
    args = ap.parse_args()

    source = json.loads(args.p0_summary.read_text(encoding="utf-8"))
    pmap = prompt_order(source["cases"])

    evaluated = [
        evaluate_case(args.image_root, pmap[c["prompt"]], c)
        for c in source["cases"]
    ]

    train = [c for c in evaluated if c["seed"] in (100, 101)]
    test = [c for c in evaluated if c["seed"] == 102]

    alphas = {
        selector: fit_global_alpha(train, selector)
        for selector in SELECTORS
    }
    attach_fixed_alpha_recovery(args.image_root, evaluated, source["cases"], alphas)

    train = [c for c in evaluated if c["seed"] in (100, 101)]
    test = [c for c in evaluated if c["seed"] == 102]

    full_agg = {selector: aggregate(evaluated, selector) for selector in SELECTORS}
    test_agg = {selector: aggregate(test, selector) for selector in SELECTORS}

    edge_capture_margin = (
        full_agg["edge_energy"]["mean_capture"]
        - full_agg["random"]["mean_capture"]
    )
    edge_test_recovery = test_agg["edge_energy"]["mean_fixed_alpha_recovery"]
    random_test_recovery = test_agg["random"]["mean_fixed_alpha_recovery"]

    claim = {
        "edge_minus_random_mean_capture": float(edge_capture_margin),
        "edge_train_fitted_alpha": float(alphas["edge_energy"]),
        "edge_test_mean_fixed_alpha_recovery": float(edge_test_recovery),
        "random_test_mean_fixed_alpha_recovery": float(random_test_recovery),
        "edge_test_positive_cases": int(test_agg["edge_energy"]["positive_fixed_alpha_cases"]),
        "selection_signal_pass": bool(edge_capture_margin >= 0.05),
        "actuation_salvage_pass": bool(
            edge_test_recovery >= 0.05
            and edge_test_recovery >= random_test_recovery + 0.03
            and test_agg["edge_energy"]["positive_fixed_alpha_cases"] >= 2
        ),
    }

    if claim["selection_signal_pass"] and claim["actuation_salvage_pass"]:
        verdict = "selection survives; P0 crop actuation was too strong but scalar blending salvages it"
    elif claim["selection_signal_pass"]:
        verdict = "selection survives; independent crop diffusion is the wrong local actuator"
    else:
        verdict = "selection itself does not survive the 4x4 practical setting"

    payload = {
        "source_gate": "Practical diffusion router P0",
        "train_seeds_for_alpha": [100, 101],
        "held_out_seed_for_alpha": 102,
        "fitted_alphas": alphas,
        "aggregate_all_cases": full_agg,
        "aggregate_held_out_seed102": test_agg,
        "claim_receipt": claim,
        "verdict": verdict,
        "cases": evaluated,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "fitted_alphas": alphas,
        "aggregate_all_cases": full_agg,
        "aggregate_held_out_seed102": test_agg,
        "claim_receipt": claim,
        "verdict": verdict,
    }, indent=2))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()

"""Gate 6: designed sparse sensing is not post-hoc sparsification.

Gate 5 designed a sparse sensing operator *before* observation. The sparse rows
therefore generated the measurements they were later used to interpret.

This gate attacks a tempting shortcut: sense densely, throw away most of the
operator interactions afterward, and hope the same sparse decoder still sees
the signal.

One local relation residual is an 8-sparse vector in R^192. A fixed dense
Gaussian operator A produces y = A x + z. For each row support d we form A_s by
zeroing all but d entries of each row of A.

Three OMP decoders get the same known sparsity k=8:

dense:
    decode (A, y)

designed sparse:
    physically observe y_s = A_s x + z
    decode (A_s, y_s)

post-hoc sparse:
    physically observe dense y = A x + z
    keep only A_s downstream
    decode (A_s, (d/p) y)

The response scaling follows the natural density correction. For OMP, positive
scalar response rescaling cannot repair a support mismatch: it rescales every
correlation/residual together, so the selected support path is unchanged.

The scientific question is deliberately finite and algorithm-specific:
does a sparse operator remain useful when it did *not* generate the response?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

P = 192
K = 8
M = 96
ROW_SUPPORTS = (16, 32, 64, 96, 192)
NOISE_SD = 0.01


@dataclass(frozen=True)
class MethodReceipt:
    mean_support_recall: float
    exact_support_fraction: float
    mean_nrmse: float


@dataclass(frozen=True)
class SupportReceipt:
    row_support: int
    density: float
    coordinate_touches: int
    dense: MethodReceipt
    designed_sparse: MethodReceipt
    posthoc_sparse: MethodReceipt


@dataclass(frozen=True)
class Gate6Receipt:
    signal_dimension: int
    signal_sparsity: int
    measurement_rows: int
    noise_sd: float
    trials: int
    row_supports: list[int]
    results: list[SupportReceipt]


def dense_operator(seed: int = 710) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(M, P)) / np.sqrt(M)


def sparsify_operator(
    dense: np.ndarray,
    row_support: int,
    seed: int,
) -> np.ndarray:
    if int(row_support) == dense.shape[1]:
        return dense.copy()

    rng = np.random.default_rng(seed)
    sparse = np.zeros_like(dense)
    for row in range(dense.shape[0]):
        support = rng.choice(
            dense.shape[1],
            int(row_support),
            replace=False,
        )
        sparse[row, support] = dense[row, support]
    return sparse


def sparse_signal(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    signal = np.zeros(P, dtype=np.float64)
    support = rng.choice(P, K, replace=False)
    signal[support] = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=K,
    )
    return signal, np.asarray(support, dtype=np.int64)


def omp(
    operator: np.ndarray,
    response: np.ndarray,
    sparsity: int = K,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(response, dtype=np.float64).copy()
    support: list[int] = []

    for _ in range(int(sparsity)):
        correlation = operator.T @ residual
        if support:
            correlation[np.asarray(support, dtype=np.int64)] = 0.0

        index = int(np.argmax(np.abs(correlation)))
        support.append(index)

        selected = operator[:, support]
        coefficient = np.linalg.lstsq(
            selected,
            response,
            rcond=None,
        )[0]
        residual = response - selected @ coefficient

    estimate = np.zeros(operator.shape[1], dtype=np.float64)
    selected = operator[:, support]
    estimate[np.asarray(support, dtype=np.int64)] = np.linalg.lstsq(
        selected,
        response,
        rcond=None,
    )[0]
    return estimate, np.asarray(support, dtype=np.int64)


def one_trial(
    trial: int,
    dense: np.ndarray,
    sparse: np.ndarray,
    row_support: int,
) -> dict[str, tuple[float, float, float]]:
    signal, true_support = sparse_signal(10000 + int(trial))
    rng = np.random.default_rng(30000 + int(trial))
    noise = rng.normal(scale=NOISE_SD, size=M)

    dense_response = dense @ signal + noise
    sparse_response = sparse @ signal + noise

    density = float(row_support) / P
    methods = {
        "dense": (dense, dense_response),
        "designed_sparse": (sparse, sparse_response),
        "posthoc_sparse": (sparse, density * dense_response),
    }

    result: dict[str, tuple[float, float, float]] = {}
    true_set = set(int(index) for index in true_support)

    for name, (operator, response) in methods.items():
        estimate, recovered_support = omp(operator, response)
        recovered_set = set(int(index) for index in recovered_support)

        recall = len(true_set & recovered_set) / K
        exact = float(true_set == recovered_set)
        nrmse = float(
            np.linalg.norm(signal - estimate)
            / np.linalg.norm(signal)
        )
        result[name] = (float(recall), exact, nrmse)

    return result


def summarize(values: list[tuple[float, float, float]]) -> MethodReceipt:
    array = np.asarray(values, dtype=np.float64)
    return MethodReceipt(
        mean_support_recall=float(array[:, 0].mean()),
        exact_support_fraction=float(array[:, 1].mean()),
        mean_nrmse=float(array[:, 2].mean()),
    )


def run_gate6(trials: int = 240) -> Gate6Receipt:
    dense = dense_operator()
    receipts: list[SupportReceipt] = []

    for row_support in ROW_SUPPORTS:
        sparse = sparsify_operator(
            dense,
            int(row_support),
            seed=9000 + int(row_support),
        )

        collected = {
            "dense": [],
            "designed_sparse": [],
            "posthoc_sparse": [],
        }
        for trial in range(int(trials)):
            outcome = one_trial(
                trial,
                dense,
                sparse,
                int(row_support),
            )
            for method in collected:
                collected[method].append(outcome[method])

        receipts.append(
            SupportReceipt(
                row_support=int(row_support),
                density=float(row_support) / P,
                coordinate_touches=M * int(row_support),
                dense=summarize(collected["dense"]),
                designed_sparse=summarize(
                    collected["designed_sparse"]
                ),
                posthoc_sparse=summarize(
                    collected["posthoc_sparse"]
                ),
            )
        )

    return Gate6Receipt(
        signal_dimension=P,
        signal_sparsity=K,
        measurement_rows=M,
        noise_sd=NOISE_SD,
        trials=int(trials),
        row_supports=[int(value) for value in ROW_SUPPORTS],
        results=receipts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=240)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate6_summary.json"),
    )
    args = parser.parse_args()

    receipt = run_gate6(args.trials)
    payload = asdict(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

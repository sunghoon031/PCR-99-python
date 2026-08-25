#!/usr/bin/env python3
"""Run PCR99c on PAIR-block TXT files and report MAA@5/10/15."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import torch

from pcr99c_registration import register_points
from txt_parser import load_registration_pairs

Tensor = torch.Tensor

def rotation_error_degrees(estimated: Tensor, ground_truth: Tensor) -> float:
    relative = estimated.transpose(0, 1).mm(ground_truth)
    cosine = (float(torch.trace(relative).item()) - 1.0) * 0.5
    cosine = min(1.0, max(-1.0, cosine))
    return math.degrees(math.acos(cosine))

def mean_average_accuracy(rotation_errors: Sequence[float], max_deg: int) -> float:
    """Mean of success rates at integer thresholds 1,...,max_deg (strict <)."""
    if not rotation_errors:
        return float("nan")
    successes = 0
    for threshold in range(1, max_deg + 1):
        successes += sum(error < threshold for error in rotation_errors)
    return successes / (len(rotation_errors) * max_deg)

def _parse_index_spec(specification: str) -> set[int]:
    indices: set[int] = set()
    if not specification.strip():
        return indices
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first < 0 or last < first:
                raise ValueError(f"invalid pair range '{item}'")
            indices.update(range(first, last + 1))
        else:
            index = int(item)
            if index < 0:
                raise ValueError("pair indices must be nonnegative")
            indices.add(index)
    return indices

def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("a CUDA device was requested, but CUDA is unavailable")
    return device

def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the optimized PyTorch PCR99c translation on PAIR-block TXT files "
            "and report rotation MAA@5/10/15."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="input TXT file(s)")
    parser.add_argument(
        "--sigmas",
        "--sigma",
        dest="sigmas",
        nargs="+",
        type=float,
        required=True,
        help="one or more PCR99c noise bounds",
    )
    parser.add_argument(
        "--thr1",
        type=float,
        required=True,
        help="log-distance-ratio truncation and prescreen threshold",
    )
    parser.add_argument(
        "--thr2",
        type=float,
        required=True,
        help="inlier-distance multiplier; threshold = sigma * thr2",
    )
    parser.add_argument(
        "--n-hypo",
        "--n_hypo",
        dest="n_hypo",
        type=int,
        required=True,
        help="number of hypotheses scored per batch",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="known source-to-target scale"
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, etc."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="float64 is closest to MATLAB; float32 is faster",
    )
    parser.add_argument(
        "--ranking-block-size",
        type=int,
        default=1024,
        help="block size for pairwise ranking costs",
    )
    parser.add_argument(
        "--triplet-chunk-size",
        type=int,
        default=65536,
        help="raw ranked triplets prescreened at once",
    )
    parser.add_argument(
        "--score-memory-mb",
        type=float,
        default=256.0,
        help="approximate temporary-memory budget for hypothesis scoring",
    )
    parser.add_argument(
        "--max-hypotheses",
        type=int,
        default=0,
        help="maximum compatible hypotheses per pair; 0 preserves no cap",
    )
    parser.add_argument(
        "--early-stop-ratio",
        type=float,
        default=0.009,
        help="MATLAB early-stop inlier ratio",
    )
    parser.add_argument(
        "--early-stop-min",
        type=int,
        default=9,
        help="minimum inlier count for early stopping",
    )
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        help="evaluate until exhaustion or --max-hypotheses",
    )
    parser.add_argument(
        "--skip-pairs",
        default="",
        metavar="SPEC",
        help="pair indices to skip, e.g. 0,4,10-15",
    )
    return parser

def run_cli(args: argparse.Namespace) -> int:
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = _resolve_device(args.device)
    skipped_pairs = _parse_index_spec(args.skip_pairs)

    print(f"Using device: {device.type.upper()}", flush=True)

    if args.early_stop_ratio < 0 or args.early_stop_min < 0:
        raise ValueError("early-stop parameters must be nonnegative")
    if any(sigma <= 0 for sigma in args.sigmas):
        raise ValueError("all sigma values must be positive")
    for input_path in args.inputs:
        if not input_path.is_file():
            raise ValueError(f"input file does not exist: {input_path}")

    pairs = []
    for input_path in args.inputs:
        pairs.extend(
            pair
            for pair in load_registration_pairs(input_path, dtype)
            if pair.pair_index not in skipped_pairs
        )

    if not pairs:
        raise ValueError("no pairs were parsed for evaluation")

    print(f"Data parsing completed: {len(pairs)} pairs.", flush=True)

    summaries: list[tuple[float, float, float, float, float]] = []

    for sigma in args.sigmas:
        print(f"Evaluating sigma = {sigma:g}", flush=True)
        rotation_errors: list[float] = []
        elapsed_times: list[float] = []

        for pair in pairs:
            source = pair.source.to(device=device, non_blocking=True)
            target = pair.target.to(device=device, non_blocking=True)
            gt_rotation = pair.gt_transform[:3, :3].to(
                device=device, non_blocking=True
            )

            _synchronize(device)
            start_time = time.perf_counter()
            result = register_points(
                source,
                target,
                sigma,
                args.thr1,
                args.thr2,
                args.n_hypo,
                args.scale,
                args.ranking_block_size,
                args.triplet_chunk_size,
                args.score_memory_mb,
                args.max_hypotheses,
                args.early_stop_ratio,
                args.early_stop_min,
                args.disable_early_stop,
            )
            _synchronize(device)
            elapsed_times.append(time.perf_counter() - start_time)

            if result.success:
                r_error = rotation_error_degrees(result.rotation, gt_rotation)
            else:
                r_error = float("inf")
            rotation_errors.append(r_error)

            del source, target, gt_rotation, result

        summaries.append(
            (
                sigma,
                mean_average_accuracy(rotation_errors, 5),
                mean_average_accuracy(rotation_errors, 10),
                mean_average_accuracy(rotation_errors, 15),
                sum(elapsed_times) / len(elapsed_times),
            )
        )

    print("sigma MAA@5 MAA@10 MAA@15 avg_time_sec")
    for sigma, maa5, maa10, maa15, average_time in summaries:
        print(
            f"{sigma:g} {maa5:.6f} {maa10:.6f} {maa15:.6f} "
            f"{average_time:.6f}"
        )
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    try:
        return run_cli(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    main()

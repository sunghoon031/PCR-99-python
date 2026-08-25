"""Parse PCR99.7 PAIR-block TXT files into PyTorch tensors."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

Tensor = torch.Tensor


@dataclass
class PairData:
    pair_index: int
    gt_transform: Tensor  # [4, 4], source -> target
    source: Tensor  # [N, 3]
    target: Tensor


_PAIR_RE = re.compile(
    r"^[^\S\r\n]*PAIR[^\S\r\n]+(\d+)\b[^\r\n]*",
    re.IGNORECASE | re.MULTILINE,
)

_FLOAT_RE = re.compile(
    r"(?<![A-Za-z_])"
    r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf(?:inity)?)"
    r"(?![A-Za-z_])",
    re.IGNORECASE,
)

_FIELD_ALIASES = {
    "gt_transform": ("gt_trans", "gt_transform"),
    "n_corr": ("n_corr",),
    "source": ("src_keypts_corr", "source_keypoints", "src_keypoints"),
    "target": ("tgt_keypts_corr", "target_keypoints", "tgt_keypoints"),
}

_ALL_FIELD_NAMES = tuple(
    alias for aliases in _FIELD_ALIASES.values() for alias in aliases
)
_FIELD_HEADER_RE = re.compile(
    rf"^[^\S\r\n]*(?P<name>{'|'.join(map(re.escape, _ALL_FIELD_NAMES))})"
    r"[^\S\r\n]*:[^\S\r\n]*",
    re.IGNORECASE | re.MULTILINE,
)


def _split_field_sections(block: str) -> dict[str, str]:
    """Split one PAIR block into complete field strings in a single pass."""
    matches = list(_FIELD_HEADER_RE.finditer(block))
    sections: dict[str, str] = {}
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(block)
        sections[match.group("name").lower()] = block[match.end():end]
    return sections


def _get_section(
    sections: dict[str, str], aliases: Sequence[str], field_name: str
) -> str:
    for alias in aliases:
        section = sections.get(alias.lower())
        if section is not None:
            return section
    joined = " or ".join(f"'{alias}:'" for alias in aliases)
    raise ValueError(f"missing required field {joined} ({field_name})")


def _read_values(section: str, expected_count: int, field_name: str) -> list[float]:
    values = [float(token) for token in _FLOAT_RE.findall(section)]
    if len(values) < expected_count:
        raise ValueError(
            f"field '{field_name}' contains {len(values)} numeric values; "
            f"expected {expected_count}"
        )
    return values[:expected_count]


def _parse_pair_block(block: str, pair_index: int, dtype: torch.dtype) -> PairData:
    sections = _split_field_sections(block)
    n_values = _read_values(
        _get_section(sections, _FIELD_ALIASES["n_corr"], "n_corr"),
        1,
        "n_corr",
    )
    n_corr_float = n_values[0]
    n_corr = int(n_corr_float)
    if n_corr <= 0 or n_corr != n_corr_float:
        raise ValueError(f"n_corr must be a positive integer, got {n_corr_float}")

    gt_values = _read_values(
        _get_section(sections, _FIELD_ALIASES["gt_transform"], "gt_trans"),
        16,
        "gt_trans",
    )
    src_values = _read_values(
        _get_section(sections, _FIELD_ALIASES["source"], "src_keypts_corr"),
        3 * n_corr,
        "src_keypts_corr",
    )
    tgt_values = _read_values(
        _get_section(sections, _FIELD_ALIASES["target"], "tgt_keypts_corr"),
        3 * n_corr,
        "tgt_keypts_corr",
    )

    gt_transform = torch.tensor(gt_values, dtype=dtype).reshape(4, 4)
    source = torch.tensor(src_values, dtype=dtype).reshape(n_corr, 3)
    target = torch.tensor(tgt_values, dtype=dtype).reshape(n_corr, 3)

    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("correspondence arrays contain NaN or infinity")
    if not bool(torch.isfinite(gt_transform).all()):
        raise ValueError("gt_trans contains NaN or infinity")

    return PairData(pair_index, gt_transform, source, target)


def load_registration_pairs(path: Path, dtype: torch.dtype) -> list[PairData]:
    """Read the complete TXT file and parse all PAIR blocks in memory."""
    text = path.read_text(encoding="utf-8", errors="replace")
    pair_headers = list(_PAIR_RE.finditer(text))

    if pair_headers:
        indexed_blocks = [
            (
                int(header.group(1)),
                text[
                    header.end():
                    pair_headers[position + 1].start()
                    if position + 1 < len(pair_headers)
                    else len(text)
                ],
            )
            for position, header in enumerate(pair_headers)
        ]
    elif text.strip():
        indexed_blocks = [(0, text)]
    else:
        return []

    pairs: list[PairData] = []
    for pair_index, block in indexed_blocks:
        try:
            pairs.append(_parse_pair_block(block, pair_index, dtype))
        except ValueError as exc:
            raise ValueError(f"{path}, PAIR {pair_index}: {exc}") from exc
    return pairs

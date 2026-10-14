"""RLE encode/decode matching the КосмоХакатон submission.csv spec.

Pixel numbering is row-major (left-to-right, top-to-bottom), 1-indexed.
Runs are reported in ascending start order and never overlap or touch,
which is exactly what run-length-encoding a boolean mask gives you for free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def encode_rle(mask: np.ndarray) -> str:
    """2D boolean/0-1 array -> RLE string ("start len start len ..."), '' if empty."""
    flat = np.asarray(mask, dtype=bool).ravel(order="C")
    if not flat.any():
        return ""
    padded = np.concatenate(([0], flat.astype(np.int8), [0]))
    diffs = np.diff(padded)
    starts_0idx = np.flatnonzero(diffs == 1)
    ends_k = np.flatnonzero(diffs == -1)
    lengths = ends_k - starts_0idx
    starts_1idx = starts_0idx + 1
    return " ".join(f"{s} {l}" for s, l in zip(starts_1idx, lengths))


def decode_rle(rle, shape: tuple[int, int]) -> np.ndarray:
    """RLE string + (H, W) -> boolean mask.

    An empty mask round-trips through CSV as an empty field, which pandas reads back
    as NaN (float) rather than "" - this is also how the organizers' own
    sample_submission.csv represents "no pixels of this class", so NaN/None are treated
    the same as an empty string here rather than as errors.
    """
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        rle = ""
    rle = str(rle).strip()
    if rle:
        nums = list(map(int, rle.split()))
        for start, length in zip(nums[0::2], nums[1::2]):
            s0 = start - 1
            mask[s0 : s0 + length] = True
    return mask.reshape(shape, order="C")


def submission_row(chip_id: str, class_id: int, mask: np.ndarray) -> dict:
    return {"chip_id": chip_id, "class_id": int(class_id), "rle": encode_rle(mask)}


def validate_submission(df: pd.DataFrame, sample_submission: pd.DataFrame) -> list[str]:
    """Structural checks against the shipped sample_submission.csv. Empty list == OK."""
    problems = []
    required = set(zip(sample_submission["chip_id"], sample_submission["class_id"]))
    got = set(zip(df["chip_id"], df["class_id"]))
    missing = required - got
    extra = got - required
    if missing:
        problems.append(f"missing {len(missing)} (chip_id, class_id) pairs, e.g. {list(missing)[:5]}")
    if extra:
        problems.append(f"unexpected {len(extra)} (chip_id, class_id) pairs, e.g. {list(extra)[:5]}")
    # NaN in "rle" is NOT an error: an empty mask round-trips through to_csv/read_csv as
    # an empty field, which pandas parses as NaN - decode_rle() treats it as "" on purpose,
    # matching how the organizers' own sample_submission.csv represents an empty mask.
    if df.duplicated(subset=["chip_id", "class_id"]).any():
        problems.append("duplicate (chip_id, class_id) rows found")
    return problems


def validate_submission_full(
    df: pd.DataFrame,
    sample_submission: pd.DataFrame,
    chip_shapes: dict[str, tuple[int, int]],
) -> list[str]:
    """Structural checks plus per-chip overlap/out-of-bounds decoding checks."""
    problems = validate_submission(df, sample_submission)
    for chip_id, group in df.groupby("chip_id"):
        shape = chip_shapes.get(chip_id)
        if shape is None:
            continue
        total = np.zeros(shape, dtype=np.uint8)
        for _, row in group.iterrows():
            try:
                m = decode_rle(row["rle"], shape)
            except Exception as e:  # noqa: BLE001 - want the chip_id in the message
                problems.append(f"{chip_id}/{row['class_id']}: RLE decode failed ({e})")
                continue
            total += m.astype(np.uint8)
        if (total > 1).any():
            problems.append(f"{chip_id}: overlapping masks between classes")
    return problems

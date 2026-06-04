"""Loading thermal-video data from disk.

The raw acquisition produces a sequence of per-frame images (PNG or CSV). The
helpers here stack those frames into a single ``(H, W, T)`` array, optionally
cropping to the region of interest.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_png_sequence(
    directory: str | Path,
    pattern: str = "*.png",
    crop: tuple[int, int, int, int] | None = None,
):
    """Load a directory of grayscale PNG frames into a ``(H, W, T)`` array.

    Parameters
    ----------
    directory:
        Folder containing the frame images.
    pattern:
        Glob pattern for the frames (sorted naturally by frame index).
    crop:
        Optional ``(row0, row1, col0, col1)`` region of interest.

    Returns
    -------
    np.ndarray
        Stacked grayscale video, shape ``(H, W, T)``.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern), key=_frame_index)
    if not files:
        raise FileNotFoundError(f"No frames matching {pattern!r} in {directory}")

    frames = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if crop is not None:
            r0, r1, c0, c1 = crop
            img = img[r0:r1, c0:c1]
        frames.append(img)
    return np.stack(frames, axis=2).astype(float)


def load_csv_sequence(
    directory: str | Path,
    pattern: str = "*.csv",
    crop: tuple[int, int, int, int] | None = None,
):
    """Load a directory of CSV temperature frames into a ``(H, W, T)`` array."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern), key=_frame_index)
    if not files:
        raise FileNotFoundError(f"No frames matching {pattern!r} in {directory}")

    frames = []
    for f in files:
        arr = pd.read_csv(f, header=None).to_numpy()
        if crop is not None:
            r0, r1, c0, c1 = crop
            arr = arr[r0:r1, c0:c1]
        frames.append(arr)
    return np.stack(frames, axis=2).astype(float)


def _frame_index(path: Path) -> int:
    """Extract a trailing integer frame index from a filename for sorting."""
    stem = path.stem
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits) if digits else 0

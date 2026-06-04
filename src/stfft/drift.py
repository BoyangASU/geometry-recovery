"""Part-drift correction via FFT phase cross-correlation.

During acquisition the imaged part can shift between frames. Each frame is
registered against a reference (typically the temporal-mean image) by locating
the peak of the circular cross-correlation in the frequency domain, then the
frame is shifted back by the estimated offset.
"""

from __future__ import annotations

import numpy as np


def circshift(image: np.ndarray, shift_row: int, shift_col: int) -> np.ndarray:
    """Circularly shift a 2D image by ``(shift_row, shift_col)`` pixels."""
    return np.roll(np.roll(image, shift_row, axis=0), shift_col, axis=1)


def estimate_shift(reference: np.ndarray, drifted: np.ndarray):
    """Estimate the integer pixel shift between ``drifted`` and ``reference``.

    Uses circular cross-correlation computed via the FFT. The returned shift
    is wrapped into ``[-H/2, H/2) x [-W/2, W/2)`` so that it represents the
    smallest displacement.

    Returns
    -------
    (int, int)
        ``(shift_row, shift_col)`` to apply to ``drifted`` to align it with
        ``reference``.
    """
    h, w = reference.shape
    fft_ref = np.fft.fft2(reference)
    fft_drift = np.fft.fft2(drifted)
    cross_corr = np.fft.ifft2(fft_ref.conjugate() * fft_drift).real
    peak = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)

    # Wrap the peak location to the nearest displacement.
    shift_row = peak[0] if peak[0] <= h // 2 else peak[0] - h
    shift_col = peak[1] if peak[1] <= w // 2 else peak[1] - w
    return int(-shift_row), int(-shift_col)


def correct_frame(reference: np.ndarray, drifted: np.ndarray):
    """Align a single drifted frame to the reference.

    Returns
    -------
    (np.ndarray, int, int)
        The aligned frame and the applied ``(shift_row, shift_col)``.
    """
    shift_row, shift_col = estimate_shift(reference, drifted)
    return circshift(drifted, shift_row, shift_col), shift_row, shift_col


def correct_video(video: np.ndarray, reference: np.ndarray | None = None):
    """Drift-correct every frame of a video against a reference image.

    Parameters
    ----------
    video:
        3D array, shape ``(H, W, T)``.
    reference:
        Reference image. Defaults to the temporal mean of ``video``.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The corrected video ``(H, W, T)`` and the per-frame shifts
        ``(T, 2)``.
    """
    if reference is None:
        reference = np.mean(video, axis=2)

    n_frames = video.shape[2]
    corrected = np.zeros_like(video)
    shifts = np.zeros((n_frames, 2), dtype=int)
    for t in range(n_frames):
        corrected[:, :, t], dr, dc = correct_frame(reference, video[:, :, t])
        shifts[t] = (dr, dc)
    return corrected, shifts

"""Image Quality Assessment (IQA) for thermal videos.

This module implements the SNR-based image quality assessment used in the
first stage of the ST-FFT pipeline. Uninformative frames (those whose SNR
falls below the SNR of the temporal-mean reference image) are filtered out
before any further processing.
"""

from __future__ import annotations

import numpy as np


def snr(image: np.ndarray, signal_pixels: np.ndarray, noise_pixels: np.ndarray) -> float:
    """Compute a signal-to-noise ratio for a single grayscale image.

    The SNR is defined as the squared mean intensity over the *signal*
    (object) region divided by the squared mean intensity over the *noise*
    (background) region.

    Parameters
    ----------
    image:
        2D grayscale image, shape ``(H, W)``.
    signal_pixels:
        Array of ``(row, col)`` coordinates belonging to the object/signal.
    noise_pixels:
        Array of ``(row, col)`` coordinates belonging to the background/noise.

    Returns
    -------
    float
        The signal-to-noise ratio.
    """
    signal_pixels = np.asarray(signal_pixels)
    noise_pixels = np.asarray(noise_pixels)

    signal_vals = image[signal_pixels[:, 0], signal_pixels[:, 1]]
    noise_vals = image[noise_pixels[:, 0], noise_pixels[:, 1]]

    signal_power = np.mean(signal_vals) ** 2
    noise_power = np.mean(noise_vals) ** 2
    return float(signal_power / noise_power)


def assess_video(video: np.ndarray, signal_pixels: np.ndarray, noise_pixels: np.ndarray):
    """Compute the per-frame SNR for an entire thermal video.

    Parameters
    ----------
    video:
        3D array of shape ``(H, W, T)`` where ``T`` is the number of frames.
    signal_pixels, noise_pixels:
        Signal and noise pixel coordinates (typically derived from the
        temporal-mean reference image via :func:`stfft.segmentation.kmeans_threshold`).

    Returns
    -------
    np.ndarray
        Per-frame SNR scores, shape ``(T,)``.
    """
    n_frames = video.shape[2]
    scores = np.zeros(n_frames)
    for t in range(n_frames):
        scores[t] = snr(video[:, :, t], signal_pixels, noise_pixels)
    return scores


def select_frames(video: np.ndarray, scores: np.ndarray, threshold: float):
    """Keep only frames whose SNR exceeds ``threshold``.

    Parameters
    ----------
    video:
        3D array of shape ``(H, W, T)``.
    scores:
        Per-frame SNR scores, shape ``(T,)``.
    threshold:
        SNR threshold. Frames with ``scores > threshold`` are retained.
        Typically the SNR of the temporal-mean reference image.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The filtered video ``(H, W, T_kept)`` and the indices of the kept
        frames ``(T_kept,)``.
    """
    keep = np.where(scores > threshold)[0]
    return video[:, :, keep], keep

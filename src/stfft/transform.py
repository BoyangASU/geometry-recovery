"""Spatial-Temporal Fast Fourier Transform (ST-FFT).

The ST-FFT is the core image-quality-improvement step. It has two stages:

1. **FFT-S (spatial)** -- a low-pass Gaussian filter is applied to each frame
   in the 2D spatial frequency domain to suppress high-frequency sensor noise.

2. **FFT-T (temporal)** -- a 1D FFT is taken along the time axis for every
   pixel; the peak magnitude of the temporal spectrum is used to build a
   single enhanced image that aggregates the informative temporal signal.
"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import fft


def gaussian_lowpass(shape, cutoff: float = 30.0) -> np.ndarray:
    """Build a centered 2D Gaussian low-pass filter.

    Parameters
    ----------
    shape:
        ``(H, W)`` of the target frames.
    cutoff:
        Standard deviation ``D0`` of the Gaussian (the larger, the gentler
        the low-pass).

    Returns
    -------
    np.ndarray
        Filter of shape ``(H, W)`` with values in ``[0, 1]``, peak at center.
    """
    h, w = shape
    rows = np.arange(h)[:, None] - h / 2
    cols = np.arange(w)[None, :] - w / 2
    d2 = rows ** 2 + cols ** 2
    return np.exp(-d2 / (2 * cutoff ** 2))


def spatial_filter(video: np.ndarray, gaussian: np.ndarray) -> np.ndarray:
    """Apply the FFT-S Gaussian low-pass filter to every frame.

    Parameters
    ----------
    video:
        3D array, shape ``(H, W, T)``.
    gaussian:
        2D Gaussian filter from :func:`gaussian_lowpass`.

    Returns
    -------
    np.ndarray
        Spatially filtered video (real magnitudes), shape ``(H, W, T)``.
    """
    n_frames = video.shape[2]
    out = np.zeros_like(video, dtype=float)
    for t in range(n_frames):
        f = np.fft.fft2(video[:, :, t])
        f_shift = np.fft.fftshift(f)
        filtered = f_shift * gaussian
        recovered = np.fft.ifft2(np.fft.ifftshift(filtered))
        out[:, :, t] = np.abs(recovered)
    return out


def temporal_peak(video: np.ndarray) -> np.ndarray:
    """Apply the FFT-T step: per-pixel temporal FFT, keep peak magnitude.

    Parameters
    ----------
    video:
        3D array, shape ``(H, W, T)`` (typically the output of
        :func:`spatial_filter`).

    Returns
    -------
    np.ndarray
        Enhanced 2D image, shape ``(H, W)``, of the per-pixel peak temporal
        spectral magnitude.
    """
    h, w, _ = video.shape
    enhanced = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            spectrum = np.abs(fft(video[i, j, :]))
            enhanced[i, j] = spectrum.max()
    return enhanced


def st_fft(video: np.ndarray, cutoff: float = 30.0):
    """Run the full ST-FFT (FFT-S followed by FFT-T).

    Parameters
    ----------
    video:
        3D array, shape ``(H, W, T)``.
    cutoff:
        Gaussian low-pass cutoff ``D0``.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The enhanced 2D image ``(H, W)`` and the intermediate spatially
        filtered video ``(H, W, T)``.
    """
    gaussian = gaussian_lowpass(video.shape[:2], cutoff)
    spatial = spatial_filter(video, gaussian)
    enhanced = temporal_peak(spatial)
    return enhanced, spatial


def normalize(image: np.ndarray) -> np.ndarray:
    """Min-max normalize an image to the ``[0, 255]`` range as ``uint8``."""
    image = image.astype(float)
    rng = image.max() - image.min()
    if rng == 0:
        return np.zeros_like(image, dtype=np.uint8)
    return np.uint8((image - image.min()) / rng * 255)

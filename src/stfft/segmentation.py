"""Intensity-based segmentation for thermal images.

A lightweight 1D k-means (Lloyd's algorithm) on pixel intensities is used to
separate the molten/object region from the cooler background. Two- and
three-cluster variants are provided; in both cases the *object* cluster (the
highest-intensity cluster) is returned.
"""

from __future__ import annotations

import numpy as np


def _kmeans_1d(image: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-3):
    """Run 1D k-means on the non-zero pixel intensities of ``image``.

    Returns an integer label map of shape ``image.shape`` (0 for the
    background/zero pixels) and the sorted cluster centers.
    """
    nonzero = image[image.nonzero()]
    if nonzero.size == 0:
        return np.zeros_like(image, dtype=int), np.array([])

    # Initialize centers spread across the intensity range.
    lo, hi = float(nonzero.min()), float(nonzero.max())
    centers = np.linspace(lo, hi, k)

    flat = image.astype(float)
    for _ in range(max_iter):
        # Assign each pixel to the nearest center.
        dists = np.abs(flat[..., None] - centers[None, None, :])
        labels = np.argmin(dists, axis=-1)

        new_centers = centers.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers[c] = flat[mask].mean()

        shift = np.abs(new_centers - centers).sum()
        centers = new_centers
        if shift < tol:
            break

    return labels, centers


def kmeans_threshold(image: np.ndarray, k: int = 2):
    """Segment the object (highest-intensity cluster) from ``image``.

    Parameters
    ----------
    image:
        2D grayscale image.
    k:
        Number of intensity clusters (2 or 3). The object is always the
        cluster with the highest center.

    Returns
    -------
    np.ndarray
        Object intensity map: original intensities where the pixel belongs
        to the object cluster, 0 elsewhere.
    """
    labels, centers = _kmeans_1d(image, k)
    if centers.size == 0:
        return np.zeros_like(image)
    object_label = int(np.argmax(centers))
    out = np.zeros_like(image, dtype=image.dtype)
    mask = labels == object_label
    out[mask] = image[mask]
    return out


def object_pixels(object_map: np.ndarray):
    """Return signal/noise pixel coordinate arrays from an object map.

    Returns
    -------
    (binary, signal_pixels, noise_pixels)
        ``binary`` is a 0/1 mask; ``signal_pixels`` and ``noise_pixels`` are
        ``(n, 2)`` arrays of ``(row, col)`` coordinates.
    """
    binary = (object_map > 0).astype(np.uint8)
    signal = np.argwhere(binary > 0)
    noise = np.argwhere(binary == 0)
    return binary, signal, noise

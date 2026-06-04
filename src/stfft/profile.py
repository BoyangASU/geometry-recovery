"""Side-view surface-profile recovery.

Complements the top-down ellipse fitting in :mod:`stfft.geometry`. The variation
of pixel intensities across frames encodes how the part's surface stabilizes
over time. This module builds a per-pixel *index matrix* (the frame index at
which each pixel peaks), samples concentric contours by shrinking the fitted
ellipse, and extracts an upper/lower surface profile that can be fit with a
polynomial / MARS-style regression.
"""

from __future__ import annotations

import numpy as np


def index_matrix(video: np.ndarray) -> np.ndarray:
    """Frame index of the per-pixel maximum intensity.

    Parameters
    ----------
    video:
        3D array, shape ``(H, W, T)`` (typically the drift-corrected,
        IQA-filtered video).

    Returns
    -------
    np.ndarray
        2D array, shape ``(H, W)``, where each entry is the frame index at
        which that pixel attains its maximum intensity.
    """
    return np.argmax(video, axis=2)


def sample_contours(
    index_mat: np.ndarray,
    center: tuple[float, float],
    ap: float,
    bp: float,
    n_steps: int = 75,
):
    """Sample concentric elliptical contours by shrinking the fitted ellipse.

    For each shrink step ``delta`` the ellipse semi-axes are reduced by
    ``delta`` and the upper/lower index values along the contour are recorded.

    Parameters
    ----------
    index_mat:
        Index matrix from :func:`index_matrix`.
    center:
        ``(x0, y0)`` ellipse center (column, row).
    ap, bp:
        Semi-major and semi-minor axes of the fitted ellipse.
    n_steps:
        Number of shrink steps (contours) to sample.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        ``x`` sampling positions, ``upper`` profile (max index per contour),
        and ``lower`` profile (min index per contour).
    """
    x0, y0 = center
    rows, cols = index_mat.shape
    yy, xx = np.mgrid[0:rows, 0:cols]

    xs, upper, lower = [], [], []
    for delta in range(n_steps):
        a = max(ap - delta, 1e-6)
        b = max(bp - delta, 1e-6)
        ring = np.sqrt(((xx - x0) / a) ** 2 + ((yy - y0) / b) ** 2) <= 1
        vals = index_mat[ring]
        if vals.size == 0:
            continue
        xs.append(delta)
        upper.append(vals.max())
        lower.append(vals.min())

    return np.array(xs), np.array(upper), np.array(lower)


def surface_profile(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Data-driven surface profile: difference between upper and lower index.

    This is not an absolute thickness but a relative descriptor of the part's
    surface contour over time.
    """
    return np.asarray(upper) - np.asarray(lower)


def fit_profile(x: np.ndarray, profile: np.ndarray, degree: int = 7):
    """Fit a polynomial to the surface profile.

    The paper uses Multivariate Adaptive Regression Splines (MARS); a degree-7
    polynomial is provided here as a dependency-free approximation. To use MARS
    instead, fit ``profile`` against ``x`` with the ``py-earth`` package.

    Returns
    -------
    (np.poly1d, np.ndarray)
        The fitted polynomial and the fitted values at ``x``.
    """
    coeffs = np.polyfit(x, profile, degree)
    poly = np.poly1d(coeffs)
    return poly, poly(x)

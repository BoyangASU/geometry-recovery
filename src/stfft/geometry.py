"""Geometric information recovery via direct ellipse fitting.

The fused-zone boundary in the enhanced thermal image is extracted as a
contour and fit to an ellipse using the numerically stable direct least
squares method of Halir and Flusser (1998). The recovered ellipse parameters
(center, semi-axes, eccentricity, orientation) provide a functional
description of the part geometry.
"""

from __future__ import annotations

import cv2
import numpy as np


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit conic coefficients ``(a, b, c, d, e, f)`` of an ellipse.

    Fits ``F(x, y) = a x^2 + b x y + c y^2 + d x + e y + f = 0`` to the
    point set ``(x, y)`` using Halir & Flusser's numerically stable direct
    least-squares algorithm.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    D1 = np.vstack([x ** 2, x * y, y ** 2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)
    M = np.linalg.inv(C) @ M
    _, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs: np.ndarray):
    """Convert conic coefficients to ellipse parameters.

    Returns
    -------
    (x0, y0, ap, bp, e, phi)
        Center ``(x0, y0)``, semi-major / semi-minor axes ``(ap, bp)``,
        eccentricity ``e`` and orientation ``phi`` (radians).
    """
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b ** 2 - a * c
    if den > 0:
        raise ValueError("coeffs do not represent an ellipse: b^2 - 4ac must be negative.")

    x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den

    num = 2 * (a * f ** 2 + c * d ** 2 + g * b ** 2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c) ** 2 + 4 * b ** 2)
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    r = (bp / ap) ** 2
    if r > 1:
        r = 1 / r
    e = np.sqrt(1 - r)

    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan((2.0 * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2
    if not width_gt_height:
        phi += np.pi / 2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, n_pts: int = 100, tmin: float = 0.0, tmax: float = 2 * np.pi):
    """Sample ``n_pts`` points on the ellipse described by ``params``."""
    x0, y0, ap, bp, e, phi = params
    t = np.linspace(tmin, tmax, n_pts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y


def extract_contour(binary: np.ndarray) -> np.ndarray:
    """Return the largest external contour of a binary mask as ``(n, 2)`` points."""
    contours, _ = cv2.findContours(
        binary.astype(np.uint8),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("No contour found in the binary mask.")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)


def recover_geometry(binary: np.ndarray):
    """Extract the boundary contour and fit an ellipse to it.

    Parameters
    ----------
    binary:
        Binary object mask (e.g. from
        :func:`stfft.segmentation.object_pixels`).

    Returns
    -------
    dict
        ``{"x0", "y0", "ap", "bp", "eccentricity", "phi", "contour"}``.
    """
    contour = extract_contour(binary)
    coeffs = fit_ellipse(contour[:, 0], contour[:, 1])
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
    return {
        "x0": x0,
        "y0": y0,
        "ap": ap,
        "bp": bp,
        "eccentricity": e,
        "phi": phi,
        "contour": contour,
    }

"""ST-FFT: Spatial-Temporal Fast Fourier Transform for manufacturing thermography.

A four-stage pipeline for thermal-video data-quality improvement and
geometric-information recovery:

1. :mod:`stfft.iqa`          -- SNR-based frame filtering (IQA)
2. :mod:`stfft.drift`        -- FFT phase cross-correlation drift correction
3. :mod:`stfft.transform`    -- the ST-FFT (FFT-S + FFT-T) quality improvement
4. :mod:`stfft.geometry`     -- ellipse-fit geometry recovery

See :func:`run_pipeline` for an end-to-end convenience wrapper.
"""

from __future__ import annotations

import numpy as np

from . import drift, geometry, io, iqa, segmentation, transform

__all__ = [
    "drift",
    "geometry",
    "io",
    "iqa",
    "segmentation",
    "transform",
    "run_pipeline",
]

__version__ = "1.0.0"


def run_pipeline(video: np.ndarray, k: int = 2, cutoff: float = 30.0, do_drift: bool = True):
    """Run the full ST-FFT pipeline on a thermal video.

    Parameters
    ----------
    video:
        Raw thermal video, shape ``(H, W, T)``.
    k:
        Number of intensity clusters for segmentation (2 or 3).
    cutoff:
        Gaussian low-pass cutoff ``D0`` for the spatial FFT.
    do_drift:
        Whether to apply drift correction after IQA.

    Returns
    -------
    dict
        ``{"enhanced", "geometry", "kept_frames", "shifts", "reference_snr"}``.
    """
    # 1. IQA: build the reference and filter frames.
    reference = np.mean(video, axis=2)
    object_map = segmentation.kmeans_threshold(reference, k=k)
    _, signal, noise = segmentation.object_pixels(object_map)
    reference_snr = iqa.snr(reference, signal, noise)

    scores = iqa.assess_video(video, signal, noise)
    filtered, kept = iqa.select_frames(video, scores, reference_snr)

    # 2. Drift correction.
    shifts = None
    if do_drift:
        filtered, shifts = drift.correct_video(filtered)

    # 3. ST-FFT quality improvement.
    enhanced, _ = transform.st_fft(filtered, cutoff=cutoff)

    # 4. Geometry recovery from the enhanced image.
    enhanced_norm = transform.normalize(enhanced)
    object_map_enh = segmentation.kmeans_threshold(enhanced_norm, k=k)
    binary, _, _ = segmentation.object_pixels(object_map_enh)
    geom = geometry.recover_geometry(binary)

    return {
        "enhanced": enhanced,
        "geometry": geom,
        "kept_frames": kept,
        "shifts": shifts,
        "reference_snr": reference_snr,
    }

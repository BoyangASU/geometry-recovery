# ST-FFT: Spatial-Temporal Fast Fourier Transform for Manufacturing Thermography

Data-quality improvement and geometric-information recovery for in-situ thermal
videos, implementing the method from:

> **Data Quality Improvement and Geometric Information Recovery for Manufacturing
> Thermography with Spatial-Temporal Fast Fourier Transform**
> Boyang Xu and Shenghan Guo, School of Manufacturing Systems and Networks,
> Arizona State University.

Thermography lets practitioners evaluate part quality *non-destructively* during
manufacturing, but raw thermal videos are often low-resolution, noisy, and
subject to part drift across frames. **ST-FFT** cleans the video and recovers a
functional (ellipse) description of the part's geometry, validated on in-situ
thermal videos of lab-based resistance spot welding (RSW).

## Method

The pipeline has four stages:

| Stage | Module | What it does |
|-------|--------|--------------|
| 1. Image Quality Assessment (IQA) | `stfft.iqa` | Scores every frame by SNR and drops frames below the temporal-mean reference SNR. |
| 2. Drift correction | `stfft.drift` | Registers each frame to the reference via FFT phase cross-correlation. |
| 3. ST-FFT quality improvement | `stfft.transform` | **FFT-S** applies a 2D Gaussian low-pass to suppress spatial noise; **FFT-T** takes a per-pixel temporal FFT and keeps the peak magnitude to fuse the informative signal into one enhanced image. |
| 4. Geometry recovery | `stfft.geometry` | Extracts the fused-zone contour and fits an ellipse (Halir & Flusser direct least squares), yielding center, semi-axes, eccentricity, and orientation. |

Intensity segmentation (`stfft.segmentation`) uses a lightweight 1D k-means
(2 or 3 clusters) on pixel intensities; the object is the highest-intensity
cluster.

## Installation

```bash
git clone https://github.com/BoyangASU/st-fft.git
cd st-fft
pip install -e .          # core package
pip install -e ".[notebook]"   # + jupyter/seaborn/tqdm for the demo
```

Requires Python ≥ 3.9. Core dependencies: NumPy, SciPy, OpenCV, pandas,
matplotlib.

## Quick start

```python
import stfft

# video: a (H, W, T) NumPy array of grayscale thermal frames
out = stfft.run_pipeline(video, k=2, cutoff=30.0, do_drift=True)

g = out["geometry"]
print(g["x0"], g["y0"])          # ellipse center
print(g["ap"], g["bp"])          # semi-major / semi-minor axes
print(g["eccentricity"])
enhanced = out["enhanced"]       # the ST-FFT enhanced image
kept = out["kept_frames"]        # indices retained by IQA
```

### Loading your own data

```python
from stfft import io

# a folder of per-frame PNGs, cropped to the region of interest
video = io.load_png_sequence("data/RSW_case1", crop=(36, 244, 84, 350))

# or CSV temperature frames
video = io.load_csv_sequence("data/RSW_case1_csv")
```

### Step-by-step control

```python
import numpy as np
import stfft

reference = np.mean(video, axis=2)
obj = stfft.segmentation.kmeans_threshold(reference, k=2)
_, signal, noise = stfft.segmentation.object_pixels(obj)

ref_snr = stfft.iqa.snr(reference, signal, noise)
scores = stfft.iqa.assess_video(video, signal, noise)
filtered, kept = stfft.iqa.select_frames(video, scores, ref_snr)

corrected, shifts = stfft.drift.correct_video(filtered)
enhanced, _ = stfft.transform.st_fft(corrected, cutoff=30.0)

binary, *_ = stfft.segmentation.object_pixels(
    stfft.segmentation.kmeans_threshold(stfft.transform.normalize(enhanced), k=2)
)
geometry = stfft.geometry.recover_geometry(binary)
```

## Demo notebook

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) runs the full pipeline on a
**synthetic thermal video**, so it works out of the box without the private
dataset. It walks through each stage with plots and ends with the one-line
`run_pipeline` wrapper.

```bash
jupyter notebook notebooks/demo.ipynb
```

## Repository layout

```
st-fft/
├── src/stfft/
│   ├── __init__.py        # public API + run_pipeline()
│   ├── io.py              # load PNG / CSV frame sequences
│   ├── iqa.py             # SNR scoring & frame selection
│   ├── segmentation.py    # 1D k-means intensity thresholding
│   ├── drift.py           # FFT phase cross-correlation drift correction
│   ├── transform.py       # FFT-S + FFT-T (the ST-FFT)
│   └── geometry.py        # ellipse fitting & geometry recovery
├── notebooks/
│   └── demo.ipynb         # runnable end-to-end demo (synthetic data)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Notes on the data

The original RSW case studies use proprietary thermal videos and are not
included here. Point `stfft.io` at your own frame directories, or use the
synthetic generator in the demo notebook as a template.

## Citation

```bibtex
@article{xu_stfft,
  title   = {Data Quality Improvement and Geometric Information Recovery for
             Manufacturing Thermography with Spatial-Temporal Fast Fourier Transform},
  author  = {Xu, Boyang and Guo, Shenghan},
  note    = {School of Manufacturing Systems and Networks, Arizona State University}
}
```

## License

MIT — see [LICENSE](LICENSE).

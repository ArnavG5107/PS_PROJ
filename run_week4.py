"""
run_week4.py
============
Week 4 — Fusion Metrics & Evaluation Framework
Image Fusion Research Programme · Computational Science Track · CSIR-CSIO

This script:
  1. Loads IR/visible source pairs and pre-fused results from Week 3 methods.
  2. Computes all 6 metrics (entropy, SF, MI, SSIM, PSNR, VIFF) for every
     method × image pair via fusion_eval.evaluate().
  3. Saves results/metric_table.csv.
  4. Runs pairwise Wilcoxon signed-rank tests; saves results/stats_test.csv.
  5. Generates a normalised radar chart; saves results/radar_chart.png.

Directory structure expected
----------------------------
    week4/
    ├── fusion_eval.py          ← importable module
    ├── run_week4.py            ← THIS FILE
    ├── data/
    │   ├── ir/                 ← IR source images  (*.png / *.jpg)
    │   ├── visible/            ← Visible source images (paired filenames)
    │   └── fused/
    │       ├── weighted_avg/   ← one fused image per pair
    │       ├── laplacian/
    │       ├── wavelet/
    │       └── guided_filter/
    └── results/                ← created automatically

If fused images are not present, the script synthesises them on-the-fly
using the same Week 3 implementations so it can be run standalone.

Usage
-----
    python run_week4.py [--data_dir DATA_DIR] [--results_dir RESULTS_DIR]
                        [--synthesise] [--n_pairs N]
"""

import argparse
import os
import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

# ── Local module ──────────────────────────────────────────────────────────────
from fusion_eval import (
    evaluate, wilcoxon_all_metrics, radar_data,
    _to_gray_float
)

# ── Optional: import Week 3 fusion functions if available ─────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'week3'))
    from week3 import (weighted_average_fusion, laplacian_pyramid_fusion,
                       wavelet_fusion, guided_filter_fusion)
    WEEK3_AVAILABLE = True
except ImportError:
    WEEK3_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback fusion implementations (standalone, no Week 3 dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def _weighted_avg(ir: np.ndarray, vis: np.ndarray,
                  w1: float = 0.5) -> np.ndarray:
    """Weighted average fusion."""
    return np.clip(w1 * ir + (1 - w1) * vis, 0.0, 1.0)


def _laplacian_pyramid(ir: np.ndarray, vis: np.ndarray,
                        levels: int = 5) -> np.ndarray:
    """
    Laplacian pyramid fusion with maximum-absolute selection at each level.

    The Gaussian pyramid base uses simple averaging; detail bands use
    max-absolute to preserve the sharpest features from either source.
    """
    def build_gaussian(img, levels):
        pyr = [img]
        for _ in range(levels):
            img = cv2.pyrDown(img)
            pyr.append(img)
        return pyr

    def build_laplacian(gaussian):
        lap = []
        for i in range(len(gaussian) - 1):
            up = cv2.pyrUp(gaussian[i + 1],
                           dstsize=(gaussian[i].shape[1],
                                    gaussian[i].shape[0]))
            lap.append(gaussian[i] - up)
        lap.append(gaussian[-1])
        return lap

    g1 = build_gaussian(ir.astype(np.float64), levels)
    g2 = build_gaussian(vis.astype(np.float64), levels)
    l1 = build_laplacian(g1)
    l2 = build_laplacian(g2)

    # Fuse: max-absolute on detail levels, average on base
    fused_lap = []
    for i in range(len(l1)):
        if i < len(l1) - 1:
            mask = np.abs(l1[i]) >= np.abs(l2[i])
            fused_lap.append(np.where(mask, l1[i], l2[i]))
        else:
            fused_lap.append(0.5 * (l1[i] + l2[i]))

    # Reconstruct
    result = fused_lap[-1]
    for i in range(len(fused_lap) - 2, -1, -1):
        result = cv2.pyrUp(result,
                           dstsize=(fused_lap[i].shape[1],
                                    fused_lap[i].shape[0]))
        result = result + fused_lap[i]

    return np.clip(result, 0.0, 1.0)


def _dwt_wavelet(ir: np.ndarray, vis: np.ndarray,
                  wavelet: str = 'sym8', level: int = 3) -> np.ndarray:
    """
    DWT wavelet fusion.
    - Approximation band  : average
    - Detail bands        : max-absolute coefficient selection
    """
    try:
        import pywt
    except ImportError:
        warnings.warn("PyWavelets not found. Falling back to weighted average.")
        return _weighted_avg(ir, vis)

    coeffs1 = pywt.wavedec2(ir,  wavelet=wavelet, level=level)
    coeffs2 = pywt.wavedec2(vis, wavelet=wavelet, level=level)

    fused_coeffs = []
    for i, (c1, c2) in enumerate(zip(coeffs1, coeffs2)):
        if i == 0:
            # Approximation: simple average
            fused_coeffs.append(0.5 * (c1 + c2))
        else:
            # Detail bands: max-absolute rule
            fused_band = []
            for b1, b2 in zip(c1, c2):
                mask = np.abs(b1) >= np.abs(b2)
                fused_band.append(np.where(mask, b1, b2))
            fused_coeffs.append(tuple(fused_band))

    result = pywt.waverec2(fused_coeffs, wavelet=wavelet)
    # Crop to original size (waverec2 may add a row/col)
    result = result[:ir.shape[0], :ir.shape[1]]
    return np.clip(result, 0.0, 1.0)


def _guided_filter_fusion(ir: np.ndarray, vis: np.ndarray,
                            radius: int = 16, eps: float = 0.01) -> np.ndarray:
    """
    Guided filter fusion.
    Weight maps derived from Laplacian saliency; refined with guided filter.
    """
    def guided_filter(I: np.ndarray, p: np.ndarray,
                      r: int, e: float) -> np.ndarray:
        """Single-channel guided filter."""
        h, w  = I.shape
        N     = cv2.boxFilter(np.ones((h, w), dtype=np.float64),
                              -1, (2*r+1, 2*r+1))
        mean_I = cv2.boxFilter(I,   -1, (2*r+1, 2*r+1)) / N
        mean_p = cv2.boxFilter(p,   -1, (2*r+1, 2*r+1)) / N
        mean_Ip= cv2.boxFilter(I*p, -1, (2*r+1, 2*r+1)) / N
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II= cv2.boxFilter(I*I, -1, (2*r+1, 2*r+1)) / N
        var_I  = mean_II - mean_I ** 2
        a      = cov_Ip / (var_I + e)
        b      = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, -1, (2*r+1, 2*r+1)) / N
        mean_b = cv2.boxFilter(b, -1, (2*r+1, 2*r+1)) / N
        return mean_a * I + mean_b

    # Laplacian-based weight maps
    lap1 = np.abs(cv2.Laplacian(ir,  cv2.CV_64F))
    lap2 = np.abs(cv2.Laplacian(vis, cv2.CV_64F))
    w1   = lap1 / (lap1 + lap2 + 1e-10)
    w2   = 1.0 - w1

    # Refine with guided filter
    w1r = guided_filter(ir,  w1, radius, eps)
    w2r = guided_filter(vis, w2, radius, eps)

    # Normalise and fuse
    total = w1r + w2r + 1e-10
    w1r   = w1r / total
    w2r   = w2r / total
    return np.clip(w1r * ir + w2r * vis, 0.0, 1.0)


# ── Map method names to fallback functions ────────────────────────────────────
FUSION_FNS = {
    'weighted_avg':  _weighted_avg,
    'laplacian':     _laplacian_pyramid,
    'wavelet':       _dwt_wavelet,
    'guided_filter': _guided_filter_fusion,
}

# If Week 3 module is available, prefer it (using correct function names)
if WEEK3_AVAILABLE:
    FUSION_FNS['weighted_avg']  = weighted_average_fusion   # was: weighted_average
    FUSION_FNS['laplacian']     = laplacian_pyramid_fusion
    FUSION_FNS['wavelet']       = wavelet_fusion            # was: dwt_wavelet_fusion
    FUSION_FNS['guided_filter'] = guided_filter_fusion


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data generator (for standalone / demo run)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_synthetic_pairs(n: int = 20,
                           h: int = 256,
                           w: int = 256) -> list:
    """
    Generate n synthetic IR/visible image pairs that mimic real fusion datasets.

    IR images    : high-contrast thermal-like appearance
    Visible imgs : natural-looking with texture and gradients

    Returns list of (ir_array, vis_array, pair_id) tuples.
    """
    rng   = np.random.default_rng(42)
    pairs = []

    for i in range(n):
        # Base scene: random blobs (simulate objects)
        scene = np.zeros((h, w), dtype=np.float64)
        for _ in range(rng.integers(3, 8)):
            cx  = rng.integers(0, w)
            cy  = rng.integers(0, h)
            rad = rng.integers(20, 60)
            yy, xx = np.ogrid[:h, :w]
            mask = ((xx - cx)**2 + (yy - cy)**2) < rad**2
            scene[mask] = rng.uniform(0.3, 1.0)

        # IR: high contrast, no texture noise
        ir_img = cv2.GaussianBlur(scene, (9, 9), 2)
        ir_img = np.clip(ir_img + rng.normal(0, 0.02, (h, w)), 0, 1)

        # Visible: lower contrast, more texture detail
        texture = rng.random((h, w)) * 0.15
        vis_img = cv2.GaussianBlur(scene * 0.6, (5, 5), 1) + texture
        vis_img = np.clip(vis_img, 0, 1)

        pairs.append((ir_img, vis_img, f'pair_{i+1:03d}'))

    print(f"  [synthetic] Generated {n} IR/visible pairs ({h}×{w})")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_image_gray(path: str) -> np.ndarray:
    """Load image as float64 grayscale in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img.astype(np.float64) / 255.0


def find_paired_images(ir_dir: str, vis_dir: str) -> list:
    """
    Find image pairs from IR and visible directories.
    Pairs are matched by filename (stem must be identical).

    Returns
    -------
    list of (ir_path, vis_path, pair_id) tuples, sorted by pair_id.
    """
    ir_files  = {Path(p).stem: p
                 for p in sorted(glob.glob(os.path.join(ir_dir,  '*.png'))) +
                           sorted(glob.glob(os.path.join(ir_dir,  '*.jpg')))}
    vis_files = {Path(p).stem: p
                 for p in sorted(glob.glob(os.path.join(vis_dir, '*.png'))) +
                           sorted(glob.glob(os.path.join(vis_dir, '*.jpg')))}

    common = sorted(set(ir_files.keys()) & set(vis_files.keys()))
    return [(ir_files[s], vis_files[s], s) for s in common]


def load_fused_or_synthesise(method: str,
                               fused_dir: str,
                               pair_id: str,
                               ir: np.ndarray,
                               vis: np.ndarray) -> np.ndarray:
    """
    Try to load a pre-fused image from disk; synthesise if not found.

    Parameters
    ----------
    method    : str     Method name (subfolder name).
    fused_dir : str     Base directory containing method subfolders.
    pair_id   : str     Image pair stem (filename without extension).
    ir        : ndarray IR source image  (float64 [0,1]).
    vis       : ndarray Visible source image.

    Returns
    -------
    np.ndarray  Fused image float64 [0,1].
    """
    for ext in ['png', 'jpg']:
        candidate = os.path.join(fused_dir, method, f'{pair_id}.{ext}')
        if os.path.exists(candidate):
            return load_image_gray(candidate)

    # Synthesise on-the-fly
    fn = FUSION_FNS.get(method)
    if fn is None:
        raise ValueError(f"Unknown fusion method '{method}' and no fused image found.")
    return fn(ir, vis)


# ═══════════════════════════════════════════════════════════════════════════════
# Radar chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_radar(normed: dict,
               metrics: list,
               save_path: str) -> None:
    """
    Plot a publication-quality radar chart with one line per method.

    Parameters
    ----------
    normed     : dict   { method: { metric: normalised_value } }
    metrics    : list   Metric names (axes).
    save_path  : str    Output PNG path.
    """
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))

    METHOD_COLORS = {
        'weighted_avg':  '#E63946',
        'laplacian':     '#2A9D8F',
        'wavelet':       '#457B9D',
        'guided_filter': '#F4A261',
    }

    for method, values_dict in normed.items():
        vals = [values_dict.get(m, 0.0) for m in metrics]
        vals += vals[:1]
        color = METHOD_COLORS.get(method, '#888888')
        ax.plot(angles, vals, 'o-', linewidth=2.0,
                color=color, label=method.replace('_', ' ').title())
        ax.fill(angles, vals, alpha=0.12, color=color)

    # Axis labels
    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [m.upper() for m in metrics],
                      fontsize=13, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                       fontsize=9, color='grey')
    ax.set_rlabel_position(30)

    ax.set_title('Fusion Method Comparison — Radar Chart\n'
                 '(All axes normalised 0–1; higher = better)',
                 fontsize=14, fontweight='bold', pad=25)

    ax.legend(loc='upper right',
              bbox_to_anchor=(1.35, 1.15),
              fontsize=11,
              framealpha=0.9)

    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Radar chart saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main(data_dir: str = 'data',
         results_dir: str = 'results',
         n_pairs: int = None,
         synthesise: bool = False) -> None:
    """
    Run the complete Week 4 evaluation pipeline.

    Parameters
    ----------
    data_dir    : root data directory (contains ir/, visible/, fused/).
    results_dir : output directory for CSV files and figures.
    n_pairs     : limit to first N pairs (None = use all).
    synthesise  : if True, always synthesise fused images (ignore fused/).
    """
    os.makedirs(results_dir, exist_ok=True)

    ir_dir    = os.path.join(data_dir, 'ir')
    vis_dir   = os.path.join(data_dir, 'visible')
    fused_dir = os.path.join(data_dir, 'fused')

    METHODS  = ['weighted_avg', 'laplacian', 'wavelet', 'guided_filter']
    METRICS  = ['entropy', 'SF', 'MI', 'SSIM', 'PSNR', 'VIFF']

    # ── 1. Find image pairs ───────────────────────────────────────────────────
    pairs = []
    if os.path.isdir(ir_dir) and os.path.isdir(vis_dir):
        pairs = find_paired_images(ir_dir, vis_dir)
        if not pairs:
            print("[WARN] No matching pairs found. Switching to synthetic mode.")
            synthesise = True
    else:
        print(f"[WARN] data/ir or data/visible not found. Using synthetic data.")
        synthesise = True

    if synthesise or not pairs:
        pairs = _make_synthetic_pairs(n=n_pairs or 20)

    if n_pairs:
        pairs = pairs[:n_pairs]

    print(f"\nWeek 4 — Fusion Evaluation Framework")
    print(f"  Image pairs  : {len(pairs)}")
    print(f"  Methods      : {METHODS}")
    print(f"  Metrics      : {METRICS}")
    print(f"  Results dir  : {results_dir}")
    print()

    # ── 2. Compute metrics per (method, image) ────────────────────────────────
    # Store both the flat CSV rows and a nested dict for statistical testing
    csv_rows = []
    # per_image_results[method][metric] = [val_pair0, val_pair1, ...]
    per_image_results = {m: {k: [] for k in METRICS} for m in METHODS}

    for idx, pair in enumerate(pairs):
        if isinstance(pair[0], np.ndarray):
            ir, vis, pair_id = pair
        else:
            ir_path, vis_path, pair_id = pair
            ir  = load_image_gray(ir_path)
            vis = load_image_gray(vis_path)

        print(f"  [{idx+1:03d}/{len(pairs)}] {pair_id}")

        for method in METHODS:
            if synthesise or not os.path.isdir(fused_dir):
                fused = FUSION_FNS[method](ir, vis)
            else:
                fused = load_fused_or_synthesise(
                    method, fused_dir, pair_id, ir, vis)

            m_vals = evaluate(fused, ir, vis)

            # Flat CSV row
            row = {'image_id': pair_id, 'method': method}
            row.update(m_vals)
            csv_rows.append(row)

            # Nested structure for Wilcoxon tests
            for k, v in m_vals.items():
                per_image_results[method][k].append(v)

    # ── 3. Save metric table ──────────────────────────────────────────────────
    metric_csv = os.path.join(results_dir, 'metric_table.csv')
    df_metrics = pd.DataFrame(csv_rows,
                              columns=['image_id', 'method'] + METRICS)
    df_metrics.to_csv(metric_csv, index=False, float_format='%.6f')
    print(f"\n  ✓ Metric table saved → {metric_csv}")
    print(f"    {len(df_metrics)} rows  ({len(pairs)} pairs × {len(METHODS)} methods)")

    # ── 4. Print mean ± std summary table ────────────────────────────────────
    print("\n  Mean ± Std per method:\n")
    print(f"  {'Method':<18}", end='')
    for m in METRICS:
        print(f"  {m:>10}", end='')
    print()
    print("  " + "-" * (18 + 12 * len(METRICS)))

    mean_results = {}
    for method in METHODS:
        print(f"  {method:<18}", end='')
        mean_results[method] = {}
        for metric in METRICS:
            vals = np.array(per_image_results[method][metric])
            mu, sd = vals.mean(), vals.std()
            mean_results[method][metric] = float(mu)
            print(f"  {mu:6.4f}±{sd:5.4f}", end='')
        print()

    # ── 5. Wilcoxon signed-rank tests ─────────────────────────────────────────
    print("\n  Running pairwise Wilcoxon signed-rank tests...")
    df_stats = wilcoxon_all_metrics(per_image_results,
                                    metrics=METRICS, alpha=0.05)
    stats_csv = os.path.join(results_dir, 'stats_test.csv')
    df_stats.to_csv(stats_csv, index=False, float_format='%.6f')
    print(f"  ✓ Statistical tests saved → {stats_csv}")

    n_sig = df_stats['significant'].sum()
    print(f"    {n_sig}/{len(df_stats)} comparisons significant at α=0.05")

    # ── 6. Radar chart ────────────────────────────────────────────────────────
    print("\n  Generating radar chart...")
    normed = radar_data(mean_results, metrics=METRICS)
    radar_path = os.path.join(results_dir, 'radar_chart.png')
    plot_radar(normed, METRICS, radar_path)

    print("\n✓ Week 4 pipeline complete.\n")
    print(f"  Files generated:")
    print(f"    {metric_csv}")
    print(f"    {stats_csv}")
    print(f"    {radar_path}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Week 4 — Fusion Metrics & Evaluation Framework')
    parser.add_argument('--data_dir',    default='data',
                        help='Root data directory (default: data/)')
    parser.add_argument('--results_dir', default='results',
                        help='Output directory   (default: results/)')
    parser.add_argument('--synthesise',  action='store_true',
                        help='Always synthesise fused images (skip data/fused/)')
    parser.add_argument('--n_pairs',     type=int, default=None,
                        help='Limit to first N pairs (default: all)')
    args = parser.parse_args()

    main(data_dir    = args.data_dir,
         results_dir = args.results_dir,
         n_pairs     = args.n_pairs,
         synthesise  = args.synthesise)
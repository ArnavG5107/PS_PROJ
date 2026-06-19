"""
week3.py
--------
Week 3 deliverable -- Classical pixel-level & multi-scale fusion.

Implements all 4 required methods from scratch and benchmarks them across
the TNO pairs in tno3,4/, producing results/metric_table.csv with columns
image_id, method, entropy, SF, MI, SSIM (one row per method x pair), plus
a representative 4-panel comparison figure and the Laplacian-pyramid
losslessness sanity check the rubric explicitly asks for.

Methods implemented (one function each, all importable):
  - weighted_average_fusion : simple pixel-level weighted average
  - laplacian_pyramid_fusion: multi-scale, max-abs rule on detail levels,
                               average on the base level
  - wavelet_fusion          : DWT decomposition, max-abs rule on detail
                               bands (H/V/D), average on approximation band
  - guided_filter_fusion    : two-scale decomposition + guided-filter
                               weight-map refinement (Li, Kang & Hu, 2013)

Metrics (also from scratch, matching the Week 3/Week 1 spec):
  - entropy_metric    : Shannon entropy of the fused image's histogram
  - spatial_frequency : Eskicioglu & Fisher row/column frequency measure
  - fusion_mi         : MI(fused, source1) + MI(fused, source2)
  - fusion_ssim       : average SSIM of fused image against each source

Usage
-----
Run on all pairs found under tno3,4 (default location), up to 100 pairs:
    python week3.py --data "tno3,4" --out results --max-pairs 100

Pick which pair gets the 4-panel comparison figure:
    python week3.py --data "tno3,4" --example-id 015

Requirements:
    pip install opencv-python numpy matplotlib pandas PyWavelets scikit-image
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# --------------------------------------------------------------------------
# Dataset discovery -- handles several common TNO download layouts
# --------------------------------------------------------------------------

def discover_pairs(root):
    """
    Find IR/visible image pairs under `root`. Supports three layouts:

      A) one subfolder per pair, each containing exactly 2 images
         (e.g. tno3,4/001/, tno3,4/002/, ... -- same convention as tno2/<id>/)
      B) exactly two subfolders (e.g. ir/, vis/) with matching file counts,
         paired by sorted order
      C) a single flat folder where each pair shares a numeric ID embedded
         in the filename (e.g. "1_ir.bmp", "1_vis.bmp")

    Returns a list of (pair_id, image_a_path, image_b_path) tuples.
    Fusion treats both sources symmetrically, so labels are generic.
    """
    root = Path(root)
    exts = ("*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])

    # Layout A: per-pair subfolders, each with exactly 2 images
    if subdirs:
        per_pair, all_two = [], True
        for d in subdirs:
            files = sorted({f for ext in exts for f in d.glob(ext)})
            if len(files) != 2:
                all_two = False
                break
            per_pair.append((d.name, files[0], files[1]))
        if all_two and per_pair:
            return per_pair

    # Layout B: exactly two subfolders, matched by sorted order
    if len(subdirs) == 2:
        files_a = sorted({f for ext in exts for f in subdirs[0].glob(ext)})
        files_b = sorted({f for ext in exts for f in subdirs[1].glob(ext)})
        if files_a and files_b and len(files_a) == len(files_b):
            return [(f"{i + 1:03d}", fa, fb)
                     for i, (fa, fb) in enumerate(zip(files_a, files_b))]

    # Layout C: flat folder, pairs identified by a shared numeric ID
    flat_files = sorted({f for ext in exts for f in root.glob(ext)})
    if flat_files:
        groups = {}
        for f in flat_files:
            m = re.search(r"(\d+)", f.stem)
            key = m.group(1) if m else f.stem
            groups.setdefault(key, []).append(f)
        pairs = [(k, v[0], v[1]) for k, v in sorted(groups.items()) if len(v) == 2]
        if pairs:
            return pairs

    raise FileNotFoundError(
        f"Could not auto-detect IR/visible pairs under {root}. Expected "
        f"per-pair subfolders, two parallel folders, or a flat folder with "
        f"shared numeric IDs in filenames -- adjust discover_pairs() if your "
        f"tno3,4 layout differs.")


def load_image_gray(path):
    """Load an image as float64 grayscale. Raises if the file can't be read."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype(np.float64)


# --------------------------------------------------------------------------
# Method 1: weighted average fusion
# --------------------------------------------------------------------------

def weighted_average_fusion(img1, img2, w1=0.5, w2=0.5):
    """Pixel-level weighted average fusion: F = w1*I1 + w2*I2."""
    fused = w1 * img1.astype(np.float64) + w2 * img2.astype(np.float64)
    return np.clip(fused, 0, 255)


# --------------------------------------------------------------------------
# Method 2: Laplacian pyramid fusion
# --------------------------------------------------------------------------

def build_gaussian_pyramid(img, levels):
    """Successive cv2.pyrDown calls; pyramid[0] is the original image."""
    pyramid = [img.astype(np.float64)]
    for _ in range(levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid):
    """Detail levels = G[i] - upsample(G[i+1]); last entry is the coarsest
    Gaussian level itself (the residual/base band)."""
    laplacian = []
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian.append(gaussian_pyramid[i] - upsampled)
    laplacian.append(gaussian_pyramid[-1])
    return laplacian


def reconstruct_from_laplacian(laplacian_pyramid):
    """Inverse of build_laplacian_pyramid: start from the base band and add
    detail levels back in, coarsest to finest."""
    img = laplacian_pyramid[-1]
    for level in reversed(laplacian_pyramid[:-1]):
        size = (level.shape[1], level.shape[0])
        img = cv2.pyrUp(img, dstsize=size) + level
    return img


def laplacian_pyramid_fusion(img1, img2, levels=4):
    """
    Multi-scale fusion via Laplacian pyramids: detail levels fused with a
    max-absolute-coefficient rule (keep whichever source has the stronger
    edge/texture response at each pixel); base level fused by averaging.
    """
    g1 = build_gaussian_pyramid(img1, levels)
    g2 = build_gaussian_pyramid(img2, levels)
    l1 = build_laplacian_pyramid(g1)
    l2 = build_laplacian_pyramid(g2)

    fused_pyramid = []
    for i in range(len(l1) - 1):  # detail levels
        mask = np.abs(l1[i]) >= np.abs(l2[i])
        fused_pyramid.append(np.where(mask, l1[i], l2[i]))
    fused_pyramid.append(0.5 * (l1[-1] + l2[-1]))  # base level: average

    fused = reconstruct_from_laplacian(fused_pyramid)
    return np.clip(fused, 0, 255)


def verify_laplacian_lossless(img, levels=4):
    """
    Rubric sanity check: decomposing and reconstructing the SAME image
    through the Laplacian pyramid (no fusion in between, i.e. both
    "sources" are identical) must return the original image up to
    floating-point rounding. Returns the max absolute pixel error.
    """
    g = build_gaussian_pyramid(img, levels)
    l = build_laplacian_pyramid(g)
    recon = reconstruct_from_laplacian(l)
    return float(np.max(np.abs(recon - img)))


# --------------------------------------------------------------------------
# Method 3: DWT wavelet fusion
# --------------------------------------------------------------------------

def wavelet_fusion(img1, img2, wavelet="db2", level=3):
    """
    DWT-based fusion: approximation (low-frequency) band fused by
    averaging; horizontal/vertical/diagonal detail bands fused by a
    maximum-absolute-coefficient rule.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    c1 = pywt.wavedec2(img1, wavelet=wavelet, level=level)
    c2 = pywt.wavedec2(img2, wavelet=wavelet, level=level)

    fused_coeffs = [0.5 * (c1[0] + c2[0])]  # approximation band: average

    for (cH1, cV1, cD1), (cH2, cV2, cD2) in zip(c1[1:], c2[1:]):
        cH = np.where(np.abs(cH1) >= np.abs(cH2), cH1, cH2)
        cV = np.where(np.abs(cV1) >= np.abs(cV2), cV1, cV2)
        cD = np.where(np.abs(cD1) >= np.abs(cD2), cD1, cD2)
        fused_coeffs.append((cH, cV, cD))

    fused = pywt.waverec2(fused_coeffs, wavelet=wavelet)
    fused = fused[:img1.shape[0], :img1.shape[1]]  # crop any DWT padding
    return np.clip(fused, 0, 255)


# --------------------------------------------------------------------------
# Method 4: guided filter fusion (Li, Kang & Hu, 2013)
# --------------------------------------------------------------------------

def guided_filter(I, p, r, eps):
    """Standard box-filter-based guided filter: filters p using guidance I."""
    I = I.astype(np.float64)
    p = p.astype(np.float64)
    ksize = (2 * r + 1, 2 * r + 1)

    mean_I = cv2.boxFilter(I, -1, ksize)
    mean_p = cv2.boxFilter(p, -1, ksize)
    corr_I = cv2.boxFilter(I * I, -1, ksize)
    corr_Ip = cv2.boxFilter(I * p, -1, ksize)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)
    return mean_a * I + mean_b


def guided_filter_fusion(img1, img2, base_r=31, base_eps=1e-2,
                          detail_r=7, detail_eps=1e-6, sal_ksize=7):
    """
    Two-scale guided filter fusion. Each source is split into a base layer
    (averaging filter) and a detail layer (residual). A saliency-based
    binary weight map is computed per source, then refined at two
    different scales by guided filtering (large radius for the base
    layer's smooth weight map, small radius for the detail layer's
    edge-preserving weight map). Base and detail layers are recombined
    using the refined, normalized weight maps.
    """
    I1, I2 = img1.astype(np.float64), img2.astype(np.float64)

    # 1) two-scale decomposition via average filtering
    B1 = cv2.boxFilter(I1, -1, (31, 31))
    B2 = cv2.boxFilter(I2, -1, (31, 31))
    D1, D2 = I1 - B1, I2 - B2

    # 2) saliency: local average of absolute Laplacian response
    def saliency(img):
        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
        return cv2.boxFilter(np.abs(lap), -1, (sal_ksize, sal_ksize))

    S1, S2 = saliency(I1), saliency(I2)

    # 3) initial binary weight maps (which source "wins" at each pixel)
    P1 = (S1 >= S2).astype(np.float64)
    P2 = 1.0 - P1

    # 4) refine weight maps at two scales via guided filtering
    W1_B = guided_filter(I1, P1, base_r, base_eps)
    W2_B = guided_filter(I2, P2, base_r, base_eps)
    W1_D = guided_filter(I1, P1, detail_r, detail_eps)
    W2_D = guided_filter(I2, P2, detail_r, detail_eps)

    # 5) normalize so weights sum to 1 at every pixel
    sum_B = W1_B + W2_B + 1e-12
    sum_D = W1_D + W2_D + 1e-12
    W1_B, W2_B = W1_B / sum_B, W2_B / sum_B
    W1_D, W2_D = W1_D / sum_D, W2_D / sum_D

    fused = (W1_B * B1 + W2_B * B2) + (W1_D * D1 + W2_D * D2)
    return np.clip(fused, 0, 255)


# --------------------------------------------------------------------------
# Metrics: entropy, spatial frequency (SF), mutual information (MI), SSIM
# --------------------------------------------------------------------------

def entropy_metric(img):
    """Shannon entropy of the image's 256-bin intensity histogram."""
    hist, _ = np.histogram(img.astype(np.uint8), bins=256, range=(0, 255))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def spatial_frequency(img):
    """Eskicioglu & Fisher spatial frequency: combined row/column gradient energy."""
    img = img.astype(np.float64)
    rf = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
    cf = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def mutual_information(img_a, img_b, bins=256):
    """MI between two images from their joint intensity histogram."""
    a = img_a.astype(np.uint8).ravel()
    b = img_b.astype(np.uint8).ravel()
    hist_2d, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 255], [0, 255]])
    pxy = hist_2d / hist_2d.sum()
    px, py = pxy.sum(axis=1), pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / px_py[nz])))


def fusion_mi(fused, img1, img2):
    """Standard fusion MI metric: MI(fused, source1) + MI(fused, source2)."""
    return mutual_information(fused, img1) + mutual_information(fused, img2)


def fusion_ssim(fused, img1, img2):
    """Average SSIM of the fused image against each of its two sources."""
    s1 = ssim(fused, img1.astype(np.float64), data_range=255)
    s2 = ssim(fused, img2.astype(np.float64), data_range=255)
    return float((s1 + s2) / 2)


def evaluate_fused(fused, img1, img2):
    """All four required metrics for one fused image, as a dict."""
    return {
        "entropy": entropy_metric(fused),
        "SF": spatial_frequency(fused),
        "MI": fusion_mi(fused, img1, img2),
        "SSIM": fusion_ssim(fused, img1, img2),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def make_four_panel_figure(fused_dict, pair_id, out_dir):
    """One 4-panel comparison figure (one panel per method) for a
    representative pair. 300 DPI, axis labels, colorbars per panel."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, fused) in zip(axes, fused_dict.items()):
        im = ax.imshow(fused, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Fusion method comparison -- pair {pair_id}")
    fig.tight_layout()
    out_path = Path(out_dir) / f"{pair_id}_4panel_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

METHODS = {
    "weighted_average": lambda i1, i2: weighted_average_fusion(i1, i2),
    "laplacian_pyramid": lambda i1, i2: laplacian_pyramid_fusion(i1, i2),
    "dwt_wavelet": lambda i1, i2: wavelet_fusion(i1, i2),
    "guided_filter": lambda i1, i2: guided_filter_fusion(i1, i2),
}


def main():
    parser = argparse.ArgumentParser(
        description="Week 3: classical pixel & multi-scale fusion benchmark on TNO")
    parser.add_argument("--data", type=str, default="tno3,4",
                         help="Folder containing TNO IR/visible pairs")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    parser.add_argument("--max-pairs", type=int, default=100,
                         help="Max number of pairs to process (default: 100)")
    parser.add_argument("--no-figures", action="store_true",
                         help="Skip generating the 4-panel comparison figures "
                              "(useful for a quick metrics-only run)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "comparison_figures"
    if not args.no_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(args.data)[: args.max_pairs]
    print(f"Found {len(pairs)} pairs under {args.data}")

    # Required rubric sanity check before running the benchmark
    sample_img = load_image_gray(pairs[0][1])
    max_err = verify_laplacian_lossless(sample_img)
    status = "PASS" if max_err < 1.0 else "CHECK -- error larger than expected"
    print(f"[sanity check] Laplacian pyramid lossless reconstruction "
          f"(identical-image test) max error: {max_err:.6f} ({status})")

    rows = []

    for pair_id, path_a, path_b in pairs:
        img1 = load_image_gray(path_a)
        img2 = load_image_gray(path_b)
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        print(f"Processing pair {pair_id} ...")
        fused_dict = {}
        for method_name, method_fn in METHODS.items():
            fused = method_fn(img1, img2)
            metrics = evaluate_fused(fused, img1, img2)
            rows.append({"image_id": pair_id, "method": method_name, **metrics})
            fused_dict[method_name] = fused

        # Generate the 4-panel comparison figure for THIS pair (i.e. for
        # every pair, not just one representative image).
        if not args.no_figures:
            fig_path = make_four_panel_figure(fused_dict, pair_id, figures_dir)
            print(f"  saved comparison figure: {fig_path}")

    df = pd.DataFrame(rows, columns=["image_id", "method", "entropy", "SF", "MI", "SSIM"])
    csv_path = out_dir / "metric_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metric table ({len(df)} rows, {len(pairs)} pairs x "
          f"{len(METHODS)} methods) to {csv_path}")

    summary = df.groupby("method")[["entropy", "SF", "MI", "SSIM"]].agg(["mean", "std"])
    summary_path = out_dir / "metric_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved per-method mean/std summary to {summary_path}")
    print("\n", summary)

    if not args.no_figures:
        print(f"\nSaved {len(pairs)} comparison figures (one per pair) to {figures_dir}")


if __name__ == "__main__":
    main()
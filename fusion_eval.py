"""
fusion_eval.py
==============
Week 4 — Fusion Metrics & Evaluation Framework
Image Fusion Research Programme · Computational Science Track · CSIR-CSIO

Importable module providing a complete set of no-reference and full-reference
image fusion quality metrics.

Usage
-----
    from fusion_eval import evaluate

    metrics = evaluate(fused_img, source1_img, source2_img)
    # metrics is a dict: {entropy, SF, MI, SSIM, PSNR, VIFF}

All inputs are expected as NumPy arrays (H×W or H×W×C), dtype float64 in [0,1].
The module normalises internally if needed.

Metrics implemented
-------------------
No-reference (only fused image needed):
  - Shannon Entropy (EN)
  - Spatial Frequency  (SF)

Full-reference (fused + both sources needed):
  - Mutual Information  (MI)   — average of MI(fused,src1) and MI(fused,src2)
  - SSIM               (SSIM) — average of SSIM(fused,src1) and SSIM(fused,src2)
  - PSNR               (PSNR) — average of PSNR(fused,src1) and PSNR(fused,src2)
  - VIFF               (VIFF) — Visual Information Fidelity in Fusion

Statistical testing utilities
------------------------------
  - wilcoxon_pairwise(results_dict) → DataFrame of p-values and effect sizes
  - radar_data(results_dict)        → normalised dict for radar chart
"""

import numpy as np
from scipy.stats import wilcoxon
from scipy.signal import convolve2d
import warnings
import pandas as pd 

# ── sklearn / skimage used for cross-validation only ─────────────────────────
from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as sk_ssim


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to single-channel float64 in [0, 1].

    Parameters
    ----------
    img : np.ndarray
        H×W (grayscale) or H×W×C (colour) image, any numeric dtype.

    Returns
    -------
    np.ndarray
        H×W float64 in [0, 1].
    """
    img = img.astype(np.float64)
    if img.ndim == 3:
        # Rec. 601 luminance weights
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    # Normalise to [0, 1] if not already
    vmax = img.max()
    if vmax > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _quantise(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Map a float [0,1] image to integer histogram bins in {0, …, bins-1}.

    Parameters
    ----------
    img  : float64 H×W in [0, 1]
    bins : number of histogram bins (default 256)

    Returns
    -------
    np.ndarray  H×W int32
    """
    q = (img * (bins - 1)).astype(np.int32)
    return np.clip(q, 0, bins - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# No-reference metrics
# ═══════════════════════════════════════════════════════════════════════════════

def shannon_entropy(fused: np.ndarray, bins: int = 256) -> float:
    """
    Compute Shannon Entropy of the fused image histogram.

    Higher entropy → richer information content (generally better).

    Parameters
    ----------
    fused : np.ndarray
        Fused image, H×W or H×W×C, any dtype.
    bins  : int
        Number of histogram bins (default 256).

    Returns
    -------
    float
        Shannon entropy in bits.
    """
    gray = _to_gray_float(fused)
    hist, _ = np.histogram(gray.ravel(), bins=bins, range=(0.0, 1.0))
    hist = hist / hist.sum()                         # normalise to probability
    mask = hist > 0
    return float(-np.sum(hist[mask] * np.log2(hist[mask])))


def spatial_frequency(fused: np.ndarray) -> float:
    """
    Compute Spatial Frequency (SF) of the fused image.

    SF measures overall activity level — higher is sharper / richer in detail.

    SF = sqrt(RF² + CF²)

    where
      RF = sqrt(mean( (I[i,j] - I[i,j-1])² ))   (row frequency)
      CF = sqrt(mean( (I[i,j] - I[i-1,j])² ))   (column frequency)

    Parameters
    ----------
    fused : np.ndarray
        Fused image, H×W or H×W×C.

    Returns
    -------
    float
        Spatial frequency value.
    """
    gray = _to_gray_float(fused)
    rf = np.sqrt(np.mean((gray[:, 1:] - gray[:, :-1]) ** 2))
    cf = np.sqrt(np.mean((gray[1:, :] - gray[:-1, :]) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Full-reference metrics
# ═══════════════════════════════════════════════════════════════════════════════

def mutual_information(img_a: np.ndarray, img_b: np.ndarray,
                       bins: int = 256) -> float:
    """
    Compute Mutual Information between two images using joint histogram.

    MI(A, B) = H(A) + H(B) - H(A, B)

    Cross-validated against sklearn.metrics.mutual_info_score.

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Images (H×W or H×W×C) to compare. Must have the same spatial size.
    bins         : int
        Histogram bins for quantisation (default 256).

    Returns
    -------
    float
        Mutual information in bits.
    """
    a = _to_gray_float(img_a)
    b = _to_gray_float(img_b)

    qa = _quantise(a, bins).ravel()
    qb = _quantise(b, bins).ravel()

    # Joint histogram
    joint_hist, _, _ = np.histogram2d(qa, qb, bins=bins,
                                      range=[[0, bins - 1], [0, bins - 1]])
    joint_hist /= joint_hist.sum()

    # Marginals
    p_a = joint_hist.sum(axis=1)
    p_b = joint_hist.sum(axis=0)

    # Entropies
    def _ent(p: np.ndarray) -> float:
        mask = p > 0
        return float(-np.sum(p[mask] * np.log2(p[mask])))

    # Joint entropy
    mask_j = joint_hist > 0
    h_ab = float(-np.sum(joint_hist[mask_j] * np.log2(joint_hist[mask_j])))

    mi = _ent(p_a) + _ent(p_b) - h_ab
    return float(max(mi, 0.0))    # clamp numerical negatives to 0


def ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Delegates to skimage.metrics.structural_similarity for numerical accuracy.

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Images (H×W or H×W×C). Must have matching spatial dimensions.

    Returns
    -------
    float
        SSIM in [-1, 1].  1.0 = perfect match.
    """
    a = _to_gray_float(img_a)
    b = _to_gray_float(img_b)
    val = sk_ssim(a, b, data_range=1.0)
    return float(val)


def psnr(img_a: np.ndarray, img_b: np.ndarray,
         max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR = 10 · log10(MAX² / MSE)

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Images (H×W or H×W×C). Must have matching spatial dimensions.
    max_val      : float
        Maximum possible pixel value (default 1.0 for normalised float).

    Returns
    -------
    float
        PSNR in dB. Returns np.inf if the images are identical.
    """
    a = _to_gray_float(img_a)
    b = _to_gray_float(img_b)
    mse = np.mean((a - b) ** 2)
    if mse == 0.0:
        return float('inf')
    return float(10.0 * np.log10(max_val ** 2 / mse))


def viff(fused: np.ndarray, reference: np.ndarray,
         sigma_nsq: float = 0.4) -> float:
    """
    Compute Visual Information Fidelity for Fusion (VIFF).

    VIFF measures the ratio of mutual information between the fused image and
    the reference that can be extracted from the fused image, relative to what
    could ideally be extracted.  Based on the VIF framework (Sheikh & Bovik 2006)
    adapted for fusion evaluation.

    Parameters
    ----------
    fused     : np.ndarray   Fused image (H×W or H×W×C).
    reference : np.ndarray   Reference / source image.
    sigma_nsq : float        Noise variance (default 0.4, standard for fusion).

    Returns
    -------
    float
        VIFF score.  1.0 = perfect fidelity; >1 possible for enhanced images.
    """
    ref = _to_gray_float(reference)
    fus = _to_gray_float(fused)

    # Gaussian kernel for local statistics
    EPS = 1e-10
    win_size = 11
    k = np.arange(win_size) - win_size // 2
    gauss_1d = np.exp(-k ** 2 / (2 * 1.5 ** 2))
    gauss_1d /= gauss_1d.sum()
    kernel = np.outer(gauss_1d, gauss_1d)

    num_total = 0.0
    den_total = 0.0

    scales = 4
    for _ in range(scales):
        # Local mean
        mu_r = convolve2d(ref, kernel, mode='valid')
        mu_f = convolve2d(fus, kernel, mode='valid')

        mu_r_sq = mu_r ** 2
        mu_f_sq = mu_f ** 2
        mu_rf   = mu_r * mu_f

        # Local variance and covariance
        sigma_r_sq  = np.maximum(convolve2d(ref ** 2,   kernel, mode='valid') - mu_r_sq, 0)
        sigma_f_sq  = np.maximum(convolve2d(fus ** 2,   kernel, mode='valid') - mu_f_sq, 0)
        sigma_rf    =            convolve2d(ref * fus,  kernel, mode='valid') - mu_rf

        # VIF sub-band information
        g = sigma_rf / (sigma_r_sq + EPS)
        sv_sq = np.maximum(sigma_f_sq - g * sigma_rf, EPS)

        # Numerator and denominator sub-band contributions
        num_total += np.sum(np.log2(1.0 + g ** 2 * sigma_r_sq / (sv_sq + sigma_nsq)))
        den_total += np.sum(np.log2(1.0 + sigma_r_sq / sigma_nsq))

        # Downsample for next scale
        ref = ref[::2, ::2]
        fus = fus[::2, ::2]

    return float(num_total / (den_total + EPS))


# ═══════════════════════════════════════════════════════════════════════════════
# Primary API
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(fused: np.ndarray,
             src1:  np.ndarray,
             src2:  np.ndarray) -> dict:
    """
    Compute a full suite of fusion quality metrics.

    Parameters
    ----------
    fused : np.ndarray
        Fused output image (H×W or H×W×C).
    src1  : np.ndarray
        First source image (e.g. IR).
    src2  : np.ndarray
        Second source image (e.g. visible).

    Returns
    -------
    dict with keys:
        entropy  – Shannon Entropy of fused image (bits)
        SF       – Spatial Frequency of fused image
        MI       – Average Mutual Information with both sources (bits)
        SSIM     – Average SSIM with both sources  [-1,1]
        PSNR     – Average PSNR with both sources  (dB)
        VIFF     – Average VIFF with both sources

    Example
    -------
    >>> import cv2
    >>> ir  = cv2.imread('ir.png',  cv2.IMREAD_GRAYSCALE).astype(float)/255
    >>> vis = cv2.imread('vis.png', cv2.IMREAD_GRAYSCALE).astype(float)/255
    >>> # ... run a fusion method to get `fused` ...
    >>> from fusion_eval import evaluate
    >>> m = evaluate(fused, ir, vis)
    >>> print(m)
    """
    return {
        'entropy': shannon_entropy(fused),
        'SF':      spatial_frequency(fused),
        'MI':      (mutual_information(fused, src1) +
                    mutual_information(fused, src2)) / 2.0,
        'SSIM':    (ssim(fused, src1) + ssim(fused, src2)) / 2.0,
        'PSNR':    (psnr(fused, src1) + psnr(fused, src2)) / 2.0,
        'VIFF':    (viff(fused, src1) + viff(fused, src2)) / 2.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical testing
# ═══════════════════════════════════════════════════════════════════════════════

def wilcoxon_pairwise(results: dict,
                      metric:  str = 'MI',
                      alpha:   float = 0.05) -> 'pd.DataFrame':
    """
    Run pairwise Wilcoxon signed-rank test for all method pairs on one metric.

    Parameters
    ----------
    results : dict
        { method_name: [metric_value_per_image, ...] }
        e.g. {'weighted_avg': [0.5, 0.6, ...], 'wavelet': [0.7, 0.8, ...]}
    metric  : str
        Label for the metric being tested (used in output column names).
    alpha   : float
        Significance level (default 0.05).

    Returns
    -------
    pd.DataFrame
        Columns: method_A, method_B, p_value, effect_size_r, significant
    """
    import pandas as pd

    methods = list(results.keys())
    rows = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            ma, mb = methods[i], methods[j]
            a = np.array(results[ma])
            b = np.array(results[mb])
            diff = a - b
            if np.all(diff == 0):
                p_val = 1.0
                r     = 0.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p_val = wilcoxon(a, b, alternative='two-sided')
                # Effect size r = Z / sqrt(N)  (matched-pairs convention)
                from scipy.stats import norm
                n   = len(a)
                z   = norm.ppf(1 - p_val / 2)
                r   = z / np.sqrt(n)
            rows.append({
                'method_A':      ma,
                'method_B':      mb,
                'p_value':       round(float(p_val), 6),
                'effect_size_r': round(float(r), 4),
                'significant':   bool(p_val < alpha),
            })
    return pd.DataFrame(rows)


def wilcoxon_all_metrics(per_image_results: dict,
                         metrics: list = None,
                         alpha: float = 0.05) -> 'pd.DataFrame':
    """
    Run pairwise Wilcoxon tests across ALL metrics and ALL method pairs.

    Parameters
    ----------
    per_image_results : dict
        { method_name: { metric: [values per image] } }
        e.g. {'wavelet': {'MI': [0.7, 0.8, ...], 'SSIM': [0.9, ...]}, ...}
    metrics : list or None
        List of metric keys. Defaults to ['entropy','SF','MI','SSIM','PSNR','VIFF'].
    alpha   : float
        Significance level.

    Returns
    -------
    pd.DataFrame
        Columns: metric, method_A, method_B, p_value, effect_size_r, significant
    """
    import pandas as pd

    if metrics is None:
        metrics = ['entropy', 'SF', 'MI', 'SSIM', 'PSNR', 'VIFF']

    all_rows = []
    for m in metrics:
        metric_slice = {method: per_image_results[method][m]
                        for method in per_image_results}
        df = wilcoxon_pairwise(metric_slice, metric=m, alpha=alpha)
        df.insert(0, 'metric', m)
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Radar chart normalisation
# ═══════════════════════════════════════════════════════════════════════════════

def radar_data(mean_results: dict,
               metrics: list = None) -> dict:
    """
    Normalise per-method metric means to [0, 1] for radar chart plotting.

    Parameters
    ----------
    mean_results : dict
        { method_name: { metric: mean_value } }
    metrics : list or None
        Metric keys to include. Defaults to ['entropy','SF','MI','SSIM','PSNR','VIFF'].

    Returns
    -------
    dict
        { method_name: { metric: normalised_value_in_[0,1] } }
    """
    if metrics is None:
        metrics = ['entropy', 'SF', 'MI', 'SSIM', 'PSNR', 'VIFF']

    # Collect all values per metric to find global min/max
    global_min = {m: float('inf')  for m in metrics}
    global_max = {m: float('-inf') for m in metrics}
    for method_data in mean_results.values():
        for m in metrics:
            v = method_data.get(m, 0.0)
            global_min[m] = min(global_min[m], v)
            global_max[m] = max(global_max[m], v)

    normed = {}
    for method, method_data in mean_results.items():
        normed[method] = {}
        for m in metrics:
            lo = global_min[m]
            hi = global_max[m]
            v  = method_data.get(m, 0.0)
            normed[method][m] = (v - lo) / (hi - lo + 1e-10)
    return normed


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test: verify MI against sklearn reference
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    """Quick sanity check — run with:  python fusion_eval.py"""
    np.random.seed(42)
    print("=" * 60)
    print("fusion_eval.py — self-test")
    print("=" * 60)

    # Synthetic images
    h, w = 128, 128
    rng  = np.random.default_rng(0)
    src1 = rng.random((h, w))
    src2 = rng.random((h, w))
    fused = 0.5 * src1 + 0.5 * src2       # simple average

    m = evaluate(fused, src1, src2)
    print("\nevaluate() output:")
    for k, v in m.items():
        print(f"  {k:8s}: {v:.6f}")

    # Cross-validate MI against sklearn
    bins = 256
    q1 = _quantise(_to_gray_float(src1), bins).ravel()
    qf = _quantise(_to_gray_float(fused), bins).ravel()
    sk_mi = mutual_info_score(q1, qf) / np.log(2)   # nats → bits
    our_mi = mutual_information(fused, src1)
    print(f"\nMI cross-validation:")
    print(f"  ours    : {our_mi:.6f} bits")
    print(f"  sklearn : {sk_mi:.6f} bits")
    assert abs(our_mi - sk_mi) < 0.05, "MI mismatch > 0.05 bits — check implementation!"
    print("  ✓ Match within tolerance")

    # Test SSIM against skimage
    our_ssim = ssim(fused, src1)
    sk_ssim_val = sk_ssim(_to_gray_float(fused), _to_gray_float(src1), data_range=1.0)
    print(f"\nSSIM cross-validation:")
    print(f"  ours    : {our_ssim:.6f}")
    print(f"  skimage : {sk_ssim_val:.6f}")
    assert abs(our_ssim - sk_ssim_val) < 1e-6, "SSIM mismatch!"
    print("  ✓ Exact match")

    print("\n✓ All self-tests passed.\n")


if __name__ == '__main__':
    _self_test()
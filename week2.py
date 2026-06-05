"""
Week 2 — Image Registration Pipeline
Image Fusion Research Programme · Phase 1: Foundations

Supports:
  - Same-modality registration (ORB + SIFT + RANSAC)
  - Cross-modal IR-Visible registration (Mutual Information maximisation via SimpleITK)

Usage:
  python registration.py --img1 imagesw2/A.bmp --img2 imagesw2/B.bmp --mode same
  python registration.py --img1 imagesw2/IR1.bmp --img2 imagesw2/V1.bmp --mode cross
  python registration.py --mode batch_cross   (runs all 5 IR/V pairs)
  python registration.py --mode batch_same    (runs A.bmp vs B.bmp with both ORB & SIFT)
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score


# ─────────────────────────────────────────────────────────────
# DATASET CONFIG
# ─────────────────────────────────────────────────────────────

DATASET_DIR = "imagesw2"

CROSS_MODAL_PAIRS = [
    ("IR1.bmp", "V1.bmp"),
    ("IR2.bmp", "V2.bmp"),
    ("IR3.bmp", "V3.bmp"),
    ("IR4.bmp", "V4.bmp"),
    ("IR5.bmp", "V5.bmp"),
]

SAME_MODAL_PAIR = ("A.bmp", "B.bmp")


# ─────────────────────────────────────────────────────────────
# UTILITY: MUTUAL INFORMATION
# ─────────────────────────────────────────────────────────────

def mutual_information(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Mutual Information between two grayscale images.
    Images are cropped to shared dimensions before computation.
    Validated against sklearn.metrics.mutual_info_score.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=256)
    mi = mutual_info_score(None, None, contingency=hist_2d)
    return mi


# ─────────────────────────────────────────────────────────────
# UTILITY: IMAGE ENHANCEMENT (CLAHE)
# ─────────────────────────────────────────────────────────────

def enhance_image(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE contrast enhancement to improve feature detection.
    Especially useful for low-contrast IR images.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# ─────────────────────────────────────────────────────────────
# UTILITY: GROUND CONTROL POINTS — MEAN REGISTRATION ERROR
# ─────────────────────────────────────────────────────────────

def compute_mre(src_pts: np.ndarray, dst_pts: np.ndarray,
                H: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute Mean Registration Error (MRE) in pixels using RANSAC inliers
    as ground control points (GCPs). Requires at least 5 inlier GCPs.

    MRE = mean Euclidean distance between projected source points
          and corresponding destination points over all inlier GCPs.
    """
    mask_bool = mask.ravel().astype(bool)
    gcp_src = src_pts[mask_bool]   # inlier source GCPs
    gcp_dst = dst_pts[mask_bool]   # corresponding destination GCPs

    if len(gcp_src) < 5:
        print(f"  [WARNING] Only {len(gcp_src)} inlier GCPs — MRE may be unreliable (need ≥ 5)")

    projected = cv2.perspectiveTransform(gcp_src, H)
    errors = np.linalg.norm(projected - gcp_dst, axis=2).ravel()
    return float(np.mean(errors)), int(len(gcp_src))


# ─────────────────────────────────────────────────────────────
# UTILITY: CHECKERBOARD VISUALISATION
# ─────────────────────────────────────────────────────────────

def create_checkerboard(img1: np.ndarray, img2: np.ndarray,
                        block_size: int = 32) -> np.ndarray:
    """
    Create a checkerboard interleaving of two grayscale images.
    Alternating blocks show alignment quality at boundaries.
    Images must be the same size.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    checker = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if ((x // block_size) + (y // block_size)) % 2 == 0:
                checker[y:y+block_size, x:x+block_size] = img1[y:y+block_size, x:x+block_size]
            else:
                checker[y:y+block_size, x:x+block_size] = img2[y:y+block_size, x:x+block_size]
    return checker


def save_checkerboard(img1: np.ndarray, img2: np.ndarray, filepath: str):
    """Save checkerboard visualisation to file at 300 DPI."""
    checker = create_checkerboard(img1, img2)
    plt.figure(figsize=(8, 6))
    plt.imshow(checker, cmap="gray")
    plt.axis("off")
    plt.title("Checkerboard Alignment Visualisation", fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# UTILITY: COLOUR OVERLAY
# ─────────────────────────────────────────────────────────────

def save_overlay(img1: np.ndarray, img2: np.ndarray, filepath: str,
                 title: str = "Overlay"):
    """
    Save a false-colour overlay: img1 → red channel, img2 → green channel.
    Misalignment appears as colour fringing; perfect alignment appears grey.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = img1[:h, :w]
    overlay[:, :, 1] = img2[:h, :w]
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# UTILITY: SIDE-BY-SIDE COMPARISON FIGURE
# ─────────────────────────────────────────────────────────────

def save_comparison_figure(img_before_overlay: np.ndarray,
                           img_after_overlay: np.ndarray,
                           checker: np.ndarray,
                           title: str, filepath: str):
    """
    Save a 3-panel figure:
      Left   — unregistered colour overlay
      Centre — registered colour overlay
      Right  — checkerboard boundary visualisation
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_before_overlay)
    axes[0].set_title("Before Registration\n(Unregistered Overlay)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(img_after_overlay)
    axes[1].set_title("After Registration\n(Registered Overlay)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(checker, cmap="gray")
    axes[2].set_title("Checkerboard\n(Alignment Quality)", fontsize=11)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# SAME-MODALITY REGISTRATION (ORB / SIFT + RANSAC)
# ─────────────────────────────────────────────────────────────

def register_same_modality(img1: np.ndarray, img2: np.ndarray,
                            method: str = "ORB",
                            pair_label: str = "pair",
                            output_dir: str = "results/same_modal") -> dict:
    """
    Register two same-modality images using feature-based methods.

    Pipeline:
      1. Detect keypoints and descriptors (ORB or SIFT)
      2. Match descriptors with BFMatcher + Lowe ratio test (0.75)
      3. Estimate homography with RANSAC (reprojection threshold 5.0 px)
      4. Compute MRE over ≥ 5 RANSAC inlier GCPs
      5. Warp img1 onto img2 coordinate frame
      6. Save: keypoints, matches, inliers, overlays, checkerboard, comparison

    Args:
        img1, img2 : grayscale uint8 images
        method     : "ORB" or "SIFT"
        pair_label : label used for saved filenames
        output_dir : directory for all output files

    Returns:
        dict with keys: method, matches, inliers, outliers, inlier_ratio,
                        mre_pixels, n_gcps, mi_before, mi_after, mi_improvement_pct
    """
    os.makedirs(output_dir, exist_ok=True)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SAME-MODALITY REGISTRATION  |  {method}  |  {pair_label}")
    print(sep)

    # ── Build detector ──────────────────────────────────────
    if method == "ORB":
        detector = cv2.ORB_create(
            nfeatures=10000, scaleFactor=1.1, nlevels=12,
            edgeThreshold=15, fastThreshold=5
        )
        norm_type = cv2.NORM_HAMMING
    else:  # SIFT
        detector = cv2.SIFT_create(
            nfeatures=5000, contrastThreshold=0.01, edgeThreshold=5
        )
        norm_type = cv2.NORM_L2

    # ── Detect and describe ─────────────────────────────────
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    print(f"  Keypoints img1 : {len(kp1)}")
    print(f"  Keypoints img2 : {len(kp2)}")

    # Save keypoint visualisations
    kp_img1 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_img2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(output_dir, f"{pair_label}_{method}_keypoints_A.png"), kp_img1)
    cv2.imwrite(os.path.join(output_dir, f"{pair_label}_{method}_keypoints_B.png"), kp_img2)

    # ── Match ────────────────────────────────────────────────
    if des1 is None or des2 is None:
        print("  [ERROR] No descriptors found.")
        return None

    bf = cv2.BFMatcher(norm_type)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for pair in knn_matches
                    if len(pair) == 2
                    for m, n in [pair]
                    if m.distance < 0.75 * n.distance]

    print(f"  Good matches   : {len(good_matches)}")

    if len(good_matches) < 10:
        print("  [ERROR] Not enough matches for homography.")
        return None

    # Save matches figure
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:100], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(output_dir, f"{pair_label}_{method}_matches.png"), match_img)

    # ── RANSAC Homography ────────────────────────────────────
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("  [ERROR] Homography estimation failed.")
        return None

    inliers = int(np.sum(mask))
    outliers = len(mask) - inliers
    inlier_ratio = inliers / len(mask)

    print(f"  Inliers        : {inliers}")
    print(f"  Outliers       : {outliers}")
    print(f"  Inlier ratio   : {inlier_ratio:.3f}")

    # ── MRE using GCPs (RANSAC inliers) ─────────────────────
    mre, n_gcps = compute_mre(src_pts, dst_pts, H, mask)
    print(f"  GCPs used      : {n_gcps}  (RANSAC inliers as GCPs)")
    print(f"  MRE            : {mre:.3f} px")

    # ── Warp ────────────────────────────────────────────────
    registered = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    cv2.imwrite(os.path.join(output_dir, f"{pair_label}_{method}_registered.png"), registered)

    # Save inlier matches
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    inlier_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(output_dir, f"{pair_label}_{method}_inliers.png"), inlier_img)

    # ── Overlays ─────────────────────────────────────────────
    img1_resized = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    h, w = img2.shape
    before_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    before_overlay[:, :, 0] = img1_resized
    before_overlay[:, :, 1] = img2

    after_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    after_overlay[:, :, 0] = registered
    after_overlay[:, :, 1] = img2

    checker = create_checkerboard(registered, img2)

    save_comparison_figure(
        before_overlay, after_overlay, checker,
        title=f"Same-Modality Registration  |  {method}  |  {pair_label}",
        filepath=os.path.join(output_dir, f"{pair_label}_{method}_comparison.png")
    )

    # ── Mutual Information ───────────────────────────────────
    mi_before = mutual_information(img1_resized, img2)
    mi_after = mutual_information(registered, img2)
    mi_improvement = ((mi_after - mi_before) / mi_before * 100) if mi_before > 1e-10 else 0.0

    print(f"  MI before      : {mi_before:.4f}")
    print(f"  MI after       : {mi_after:.4f}")
    print(f"  MI improvement : {mi_improvement:+.2f}%")

    return {
        "pair":             pair_label,
        "method":           method,
        "matches":          len(good_matches),
        "inliers":          inliers,
        "outliers":         outliers,
        "inlier_ratio":     round(inlier_ratio, 4),
        "n_gcps":           n_gcps,
        "mre_pixels":       round(mre, 4),
        "mi_before":        round(mi_before, 4),
        "mi_after":         round(mi_after, 4),
        "mi_improvement_pct": round(mi_improvement, 2),
    }


# ─────────────────────────────────────────────────────────────
# CROSS-MODAL REGISTRATION (MI MAXIMISATION via SimpleITK)
# ─────────────────────────────────────────────────────────────

def register_cross_modal(ir_path: str, vis_path: str,
                         pair_label: str = "pair",
                         output_dir: str = "results/cross_modal") -> dict:
    """
    Register an IR image to a Visible image using Mutual Information
    maximisation (Mattes MI metric, Euler2D transform, RANSAC-free).

    Pipeline:
      1. Load images as SimpleITK Float32
      2. Geometry-centred initialisation (Euler2DTransform)
      3. Mattes MI metric, Regular Step Gradient Descent optimiser
      4. Resample moving (IR) onto fixed (Visible) grid
      5. Compute MRE via ORB GCPs on the registered pair (≥ 5 points)
      6. Save: before/after overlays, checkerboard, comparison figure

    Args:
        ir_path, vis_path : file paths to IR and Visible images
        pair_label        : label for output files
        output_dir        : directory for all output files

    Returns:
        dict with keys: pair, mi_before, mi_after,
                        mre_pixels, n_gcps (from post-registration ORB GCPs)
    """
    os.makedirs(output_dir, exist_ok=True)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  CROSS-MODAL REGISTRATION  |  MI Maximisation  |  {pair_label}")
    print(sep)

    # ── Load ─────────────────────────────────────────────────
    fixed   = sitk.ReadImage(vis_path,  sitk.sitkFloat32)   # Visible = fixed
    moving  = sitk.ReadImage(ir_path,   sitk.sitkFloat32)   # IR = moving

    # ── Registration ─────────────────────────────────────────
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300
    )
    init_tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg.SetInitialTransform(init_tx, inPlace=False)

    transform = reg.Execute(fixed, moving)

    registered_sitk = sitk.Resample(
        moving, fixed, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )

    # ── Convert to NumPy ─────────────────────────────────────
    vis_np  = sitk.GetArrayFromImage(fixed).astype(np.uint8)
    ir_np   = sitk.GetArrayFromImage(moving).astype(np.uint8)
    reg_np  = sitk.GetArrayFromImage(registered_sitk).astype(np.uint8)

    # ── MI before / after ────────────────────────────────────
    mi_before = mutual_information(vis_np, ir_np)
    mi_after  = mutual_information(vis_np, reg_np)
    mi_improvement = ((mi_after - mi_before) / mi_before * 100) if mi_before > 1e-10 else 0.0

    print(f"  MI before      : {mi_before:.4f}")
    print(f"  MI after       : {mi_after:.4f}")
    print(f"  MI improvement : {mi_improvement:+.2f}%")

    # ── MRE via ORB GCPs on registered pair ──────────────────
    # After MI registration, use ORB to find corresponding points
    # between registered IR and Visible to measure residual error.
    mre, n_gcps = _compute_mre_orb(reg_np, vis_np)
    print(f"  GCPs used      : {n_gcps}  (ORB inliers post-registration)")
    print(f"  MRE            : {mre:.3f} px")

    # ── Overlays ─────────────────────────────────────────────
    h = min(vis_np.shape[0], ir_np.shape[0])
    w = min(vis_np.shape[1], ir_np.shape[1])

    before_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    before_overlay[:, :, 0] = ir_np[:h, :w]
    before_overlay[:, :, 1] = vis_np[:h, :w]

    after_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    after_overlay[:, :, 0] = reg_np[:h, :w]
    after_overlay[:, :, 1] = vis_np[:h, :w]

    checker = create_checkerboard(reg_np[:h, :w], vis_np[:h, :w])

    # Save individual files
    save_overlay(ir_np,   vis_np, os.path.join(output_dir, f"{pair_label}_before_overlay.png"),
                 title=f"{pair_label} — Before Registration")
    save_overlay(reg_np,  vis_np, os.path.join(output_dir, f"{pair_label}_after_overlay.png"),
                 title=f"{pair_label} — After Registration")
    save_checkerboard(reg_np[:h, :w], vis_np[:h, :w],
                      os.path.join(output_dir, f"{pair_label}_checkerboard.png"))

    save_comparison_figure(
        before_overlay, after_overlay, checker,
        title=f"Cross-Modal Registration (IR→Visible)  |  {pair_label}",
        filepath=os.path.join(output_dir, f"{pair_label}_comparison.png")
    )

    return {
        "pair":               pair_label,
        "mi_before":          round(mi_before, 4),
        "mi_after":           round(mi_after, 4),
        "mi_improvement_pct": round(mi_improvement, 2),
        "n_gcps":             n_gcps,
        "mre_pixels":         round(mre, 4),
    }


def _compute_mre_orb(img_reg: np.ndarray, img_ref: np.ndarray) -> tuple:
    """
    Internal helper: estimate MRE on a registered pair using ORB + RANSAC.
    ORB inliers after registration serve as GCPs to measure residual pixel error.
    Returns (mre_pixels, n_gcps).
    """
    detector = cv2.ORB_create(nfeatures=5000, scaleFactor=1.1, nlevels=12,
                               edgeThreshold=15, fastThreshold=5)

    kp1, des1 = detector.detectAndCompute(enhance_image(img_reg), None)
    kp2, des2 = detector.detectAndCompute(enhance_image(img_ref), None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return 999.0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]

    if len(good) < 5:
        return 999.0, len(good)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return 999.0, 0

    mre, n_gcps = compute_mre(src_pts, dst_pts, H, mask)
    return mre, n_gcps


# ─────────────────────────────────────────────────────────────
# BATCH RUNNERS
# ─────────────────────────────────────────────────────────────

def run_batch_cross_modal():
    """Run MI registration on all 5 IR/Visible pairs and save CSV summary."""
    results = []
    for idx, (ir_name, vis_name) in enumerate(CROSS_MODAL_PAIRS):
        ir_path  = os.path.join(DATASET_DIR, ir_name)
        vis_path = os.path.join(DATASET_DIR, vis_name)
        label    = f"pair{idx + 1}"
        result   = register_cross_modal(ir_path, vis_path, pair_label=label)
        if result:
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        csv_path = "results/cross_modal/mi_results.csv"
        df.to_csv(csv_path, index=False)
        _print_table("CROSS-MODAL REGISTRATION SUMMARY", df)
        print(f"\n  CSV saved → {csv_path}")


def run_batch_same_modal(img1_path: str, img2_path: str):
    """Run ORB and SIFT registration on a single same-modality pair."""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"[ERROR] Could not load images: {img1_path}, {img2_path}")
        return

    print(f"\n  img1 shape : {img1.shape}")
    print(f"  img2 shape : {img2.shape}")

    img1 = enhance_image(img1)
    img2 = enhance_image(img2)

    results = []
    for method in ["ORB", "SIFT"]:
        r = register_same_modality(img1, img2, method=method,
                                   pair_label="AB", output_dir="results/same_modal")
        if r:
            results.append(r)

    if results:
        df = pd.DataFrame(results)
        csv_path = "results/same_modal/registration_results.csv"
        os.makedirs("results/same_modal", exist_ok=True)
        df.to_csv(csv_path, index=False)
        _print_table("SAME-MODALITY REGISTRATION SUMMARY", df)
        print(f"\n  CSV saved → {csv_path}")


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def _print_table(title: str, df: pd.DataFrame):
    """Pretty-print a summary table to stdout."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Week 2 — Image Registration Pipeline"
    )
    parser.add_argument(
        "--img1", type=str, default=None,
        help="Path to first image (source / IR)"
    )
    parser.add_argument(
        "--img2", type=str, default=None,
        help="Path to second image (reference / Visible)"
    )
    parser.add_argument(
        "--mode", type=str,
        choices=["same", "cross", "batch_cross", "batch_same"],
        default="batch_cross",
        help=(
            "same         → single same-modality pair (ORB + SIFT)\n"
            "cross        → single cross-modal pair (MI registration)\n"
            "batch_cross  → all 5 IR/V pairs in imagesw2/ (default)\n"
            "batch_same   → same-modal A.bmp vs B.bmp with ORB & SIFT"
        )
    )
    parser.add_argument(
        "--method", type=str, choices=["ORB", "SIFT"], default="ORB",
        help="Feature detector for same-modality mode (default: ORB)"
    )
    args = parser.parse_args()

    os.makedirs("results/same_modal",  exist_ok=True)
    os.makedirs("results/cross_modal", exist_ok=True)

    if args.mode == "batch_cross":
        run_batch_cross_modal()

    elif args.mode == "batch_same":
        img1_path = args.img1 or os.path.join(DATASET_DIR, SAME_MODAL_PAIR[0])
        img2_path = args.img2 or os.path.join(DATASET_DIR, SAME_MODAL_PAIR[1])
        run_batch_same_modal(img1_path, img2_path)

    elif args.mode == "same":
        if not args.img1 or not args.img2:
            parser.error("--img1 and --img2 are required for --mode same")
        img1 = enhance_image(cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE))
        img2 = enhance_image(cv2.imread(args.img2, cv2.IMREAD_GRAYSCALE))
        r = register_same_modality(img1, img2, method=args.method,
                                   pair_label="custom", output_dir="results/same_modal")
        if r:
            _print_table("RESULT", pd.DataFrame([r]))

    elif args.mode == "cross":
        if not args.img1 or not args.img2:
            parser.error("--img1 and --img2 are required for --mode cross")
        r = register_cross_modal(args.img1, args.img2,
                                 pair_label="custom", output_dir="results/cross_modal")
        if r:
            _print_table("RESULT", pd.DataFrame([r]))


if __name__ == "__main__":
    main()

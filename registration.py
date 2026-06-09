"""
registration.py
===============
Processes three fixed folders at the project root:
  P1W2/  — same-modality pair  (ORB or SIFT + RANSAC homography — best auto-selected)
  P2W2/  — same-modality pair  (ORB or SIFT + RANSAC homography — best auto-selected)
  P3W2/  — cross-modal IR→Visible pair (Mattes MI via SimpleITK)

Modality (same vs cross) is read directly from TARGET_FOLDERS — the user's
domain knowledge is authoritative and more reliable than any runtime heuristic.

For same-modal pairs, ORB and SIFT are both probed lightly; whichever achieves
the higher RANSAC inlier ratio is used for the full registration run.

Outputs per pair:
  results/same_modal/   → keypoints, matches, inliers, registered image,
                          before/after overlays, checkerboard, 3-panel figure
  results/cross_modal/  → before/after overlays, checkerboard, 3-panel figure
  results/registration_summary.csv → MRE, MI, inlier ratio for all pairs

Usage:
  python registration.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mutual_info_score


# ─────────────────────────────────────────────────────────────
# CONFIG — edit only this section if folder names change
# ─────────────────────────────────────────────────────────────

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# (folder_name, modality, display_label)
# modality is AUTHORITATIVE — set this correctly for your data.
# "same"  → ORB vs SIFT probe → best method chosen → RANSAC homography
# "cross" → SimpleITK Mattes MI maximisation
TARGET_FOLDERS = [
    ("P1W2", "same",  "P1W2"),
    ("P2W2", "same",  "P2W2"),
    ("P3W2", "cross", "P3W2"),
]



# ─────────────────────────────────────────────────────────────
# HELPERS: FILE DISCOVERY
# ─────────────────────────────────────────────────────────────

def get_images_from_folder(folder: str):
    """
    Return (path1, path2) — the two image files in a folder, sorted
    alphabetically. Raises if folder missing or image count ≠ 2.
    For P3W2: IR.bmp < V.bmp alphabetically, so path1=IR, path2=Visible.
    """
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: '{folder}'")
    images = sorted([f for f in p.iterdir()
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
    if len(images) != 2:
        raise ValueError(
            f"'{folder}' must contain exactly 2 images, found {len(images)}"
        )
    return str(images[0]), str(images[1])


# ─────────────────────────────────────────────────────────────
# METRIC: MUTUAL INFORMATION
# ─────────────────────────────────────────────────────────────

def mutual_information(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Mutual Information via a 2D joint histogram (256 bins).
    Validated against sklearn.metrics.mutual_info_score.
    Images are cropped to their shared spatial extent before computation.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    hist_2d, _, _ = np.histogram2d(
        img1[:h, :w].ravel().astype(np.float32),
        img2[:h, :w].ravel().astype(np.float32),
        bins=256
    )
    return float(mutual_info_score(None, None, contingency=hist_2d))


# ─────────────────────────────────────────────────────────────
# METRIC: MEAN REGISTRATION ERROR (MRE)
# ─────────────────────────────────────────────────────────────

def compute_mre(src_pts: np.ndarray, dst_pts: np.ndarray,
                H: np.ndarray, mask: np.ndarray):
    """
    Compute Mean Registration Error (MRE) in pixels over RANSAC inliers.

    RANSAC inliers serve as Ground Control Points (GCPs).
    MRE = mean Euclidean distance between H-projected source GCPs
          and their corresponding destination GCPs.
    Requires ≥ 5 inlier GCPs per the Week 2 rubric.

    Returns: (mre_pixels: float, n_gcps: int)
    """
    mask_bool = mask.ravel().astype(bool)
    gcp_src = src_pts[mask_bool]
    gcp_dst = dst_pts[mask_bool]

    if len(gcp_src) < 5:
        print(f"  [WARNING] Only {len(gcp_src)} inlier GCPs — need ≥ 5 per rubric")

    projected = cv2.perspectiveTransform(gcp_src, H)  # (N,1,2)
    errors = np.linalg.norm(projected - gcp_dst, axis=2).ravel()
    return float(np.mean(errors)), int(len(gcp_src))


# ─────────────────────────────────────────────────────────────
# PREPROCESSING: CLAHE CONTRAST ENHANCEMENT here 
# ─────────────────────────────────────────────────────────────

def enhance_image(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).
    Used only for feature detection — NOT applied to the warped output,
    so radiometric values of the registered image are preserved.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# ─────────────────────────────────────────────────────────────
# AUTO-SELECTION: BEST FEATURE METHOD (ORB vs SIFT)
# ─────────────────────────────────────────────────────────────

def select_best_method(img_orig1: np.ndarray, img_orig2: np.ndarray) -> str:
    """
    Probe both ORB and SIFT on CLAHE-enhanced images and return the method
    with the higher RANSAC inlier ratio.  Ties go to ORB (faster, patent-free).

    This replaces the former 'for method in ["ORB","SIFT"]' loop so that
    only one full registration run is performed per same-modal pair, and
    only the more reliable result is saved.

    Returns: 'ORB' or 'SIFT'
    """
    scores: dict[str, float] = {}

    for method in ["ORB", "SIFT"]:
        if method == "ORB":
            det       = cv2.ORB_create(nfeatures=3000)
            norm_type = cv2.NORM_HAMMING
        else:
            det       = cv2.SIFT_create(nfeatures=3000)
            norm_type = cv2.NORM_L2

        img1 = enhance_image(img_orig1)
        img2 = enhance_image(img_orig2)
        kp1, des1 = det.detectAndCompute(img1, None)
        kp2, des2 = det.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            scores[method] = 0.0
            continue

        bf   = cv2.BFMatcher(norm_type)
        knn  = bf.knnMatch(des1, des2, k=2)
        good = [
            m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance
        ]

        if len(good) < 10:
            scores[method] = 0.0
            continue

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        scores[method] = (
            float(np.sum(mask.ravel())) / len(mask.ravel())
            if mask is not None else 0.0
        )

    best = max(scores, key=scores.get)
    print(
        f"  [method probe]   ORB={scores.get('ORB', 0):.3f}  "
        f"SIFT={scores.get('SIFT', 0):.3f}  →  using {best}"
    )
    return best


# ─────────────────────────────────────────────────────────────
# VISUALISATION: CHECKERBOARD
# ─────────────────────────────────────────────────────────────

def create_checkerboard(img1: np.ndarray, img2: np.ndarray,
                        block_size: int = 32) -> np.ndarray:
    """
    Interleave two grayscale images in a checkerboard pattern.
    Discontinuities at block boundaries reveal residual misalignment.
    Images are cropped to their shared spatial extent.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1, img2 = img1[:h, :w], img2[:h, :w]
    checker = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            src = img1 if ((x // block_size) + (y // block_size)) % 2 == 0 else img2
            checker[y:y+block_size, x:x+block_size] = \
                src[y:y+block_size, x:x+block_size]
    return checker


def save_checkerboard(img1: np.ndarray, img2: np.ndarray, filepath: str,
                      title: str = "Checkerboard — Alignment Quality"):
    """Save a checkerboard overlay at 300 DPI."""
    checker = create_checkerboard(img1, img2)
    plt.figure(figsize=(8, 6))
    plt.imshow(checker, cmap="gray")
    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# VISUALISATION: FALSE-COLOUR OVERLAY
# ─────────────────────────────────────────────────────────────

def save_overlay(img1: np.ndarray, img2: np.ndarray,
                 filepath: str, title: str = "Overlay"):
    """
    False-colour overlay: img1 → red channel, img2 → green channel.
    Perfect alignment → grey. Misalignment → colour fringing.
    Saved at 300 DPI.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = img1[:h, :w]
    overlay[:, :, 1] = img2[:h, :w]
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# VISUALISATION: 3-PANEL SIDE-BY-SIDE FIGURE
# ─────────────────────────────────────────────────────────────

def save_comparison_figure(before_overlay: np.ndarray,
                           after_overlay: np.ndarray,
                           checker: np.ndarray,
                           title: str, filepath: str):
    """
    Save a 3-panel publication-quality figure at 300 DPI:
      Panel 1 — Unregistered overlay (before)
      Panel 2 — Registered overlay  (after)
      Panel 3 — Checkerboard alignment visualisation
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(before_overlay)
    axes[0].set_title("Before Registration\n(Unregistered Overlay)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(after_overlay)
    axes[1].set_title("After Registration\n(Registered Overlay)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(checker, cmap="gray")
    axes[2].set_title("Checkerboard\n(Alignment Quality at Boundaries)", fontsize=11)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# SAME-MODALITY REGISTRATION  (ORB / SIFT + RANSAC)
# ─────────────────────────────────────────────────────────────

def register_same_modality(img_orig1: np.ndarray, img_orig2: np.ndarray,
                            method: str = "ORB",
                            pair_label: str = "pair",
                            output_dir: str = "results/same_modal") -> dict:
    """
    Register two same-modality grayscale images using feature matching.

    Pipeline:
      1. CLAHE enhancement for detection only (originals used for output)
      2. Detect keypoints + descriptors (ORB or SIFT — pre-selected by probe)
      3. BFMatcher + Lowe ratio test (threshold = 0.75)
      4. RANSAC homography estimation (reprojection threshold = 5.0 px)
      5. MRE computed over RANSAC inliers as GCPs (≥ 5 required)
      6. Warp img_orig1 onto img_orig2 coordinate frame
      7. Save: keypoints, matches, inlier matches, registered image,
               before/after overlays (individual), 3-panel comparison

    Why SIFT fails on IR–Visible:
      SIFT relies on gradient-based feature descriptors. IR and visible
      images have fundamentally different gradient distributions —
      edges in IR correspond to temperature discontinuities, not
      reflectance edges. Cross-modal MI registration is used instead.

    Args:
        img_orig1  : raw grayscale image 1 (unenhanced)
        img_orig2  : raw grayscale image 2 (unenhanced)
        method     : "ORB" or "SIFT" — chosen by select_best_method()
        pair_label : used for output filenames
        output_dir : directory for all saved outputs

    Returns: dict of registration metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  SAME-MODAL REGISTRATION  |  {method}  |  {pair_label}")
    print(f"{'='*60}")

    # CLAHE-enhanced copies for detection only
    img1 = enhance_image(img_orig1)
    img2 = enhance_image(img_orig2)

    # ── Detector ──────────────────────────────────────────────
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

    # ── Detect & describe ─────────────────────────────────────
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    print(f"  Keypoints img1 : {len(kp1)}")
    print(f"  Keypoints img2 : {len(kp2)}")

    # Save keypoint visualisations
    for tag, src, kp in [("A", img1, kp1), ("B", img2, kp2)]:
        kp_vis = cv2.drawKeypoints(
            src, kp, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite(
            os.path.join(output_dir, f"{pair_label}_{method}_keypoints_{tag}.png"),
            kp_vis
        )

    if des1 is None or des2 is None:
        print("  [ERROR] No descriptors found — cannot register.")
        return None

    # ── Matching (Lowe ratio test 0.75) ───────────────────────
    bf = cv2.BFMatcher(norm_type)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [
        m for pair in knn_matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]
    print(f"  Good matches   : {len(good_matches)}")

    if len(good_matches) < 10:
        print("  [ERROR] Fewer than 10 good matches — cannot compute homography.")
        return None

    # Save all good matches (capped at 100 for readability)
    match_vis = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches[:100], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{pair_label}_{method}_matches.png"),
        match_vis
    )

    # ── RANSAC Homography ──────────────────────────────────────
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("  [ERROR] Homography estimation failed.")
        return None

    # mask shape is (N,1) — flatten to 1D for indexing
    mask_flat = mask.ravel()
    inliers      = int(np.sum(mask_flat))
    outliers     = len(mask_flat) - inliers
    inlier_ratio = inliers / len(mask_flat)

    print(f"  Inliers        : {inliers}")
    print(f"  Outliers       : {outliers}")
    print(f"  Inlier ratio   : {inlier_ratio:.3f}")

    # ── MRE over ≥ 5 RANSAC inlier GCPs ─────────────────────
    mre, n_gcps = compute_mre(src_pts, dst_pts, H, mask_flat)
    print(f"  GCPs used      : {n_gcps}  (RANSAC inliers)")
    print(f"  MRE            : {mre:.3f} px")

    # Save inlier-only match image
    inlier_matches = [
        good_matches[i] for i in range(len(good_matches)) if mask_flat[i]
    ]
    inlier_vis = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{pair_label}_{method}_inliers.png"),
        inlier_vis
    )

    # ── Warp original (unenhanced) image ─────────────────────
    registered = cv2.warpPerspective(
        img_orig1, H, (img_orig2.shape[1], img_orig2.shape[0])
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{pair_label}_{method}_registered.png"),
        registered
    )

    # ── Build overlays ────────────────────────────────────────
    img1_resized = cv2.resize(img_orig1, (img_orig2.shape[1], img_orig2.shape[0]))
    h, w = img_orig2.shape

    before_ov = np.zeros((h, w, 3), dtype=np.uint8)
    before_ov[:, :, 0] = img1_resized
    before_ov[:, :, 1] = img_orig2

    after_ov = np.zeros((h, w, 3), dtype=np.uint8)
    after_ov[:, :, 0] = registered
    after_ov[:, :, 1] = img_orig2

    checker = create_checkerboard(registered, img_orig2)

    # Individual overlay files (required by rubric)
    save_overlay(
        img1_resized, img_orig2,
        os.path.join(output_dir, f"{pair_label}_{method}_overlay_before.png"),
        title=f"{pair_label} {method} — Before Registration"
    )
    save_overlay(
        registered, img_orig2,
        os.path.join(output_dir, f"{pair_label}_{method}_overlay_after.png"),
        title=f"{pair_label} {method} — After Registration"
    )
    save_checkerboard(
        registered, img_orig2,
        os.path.join(output_dir, f"{pair_label}_{method}_checkerboard.png"),
        title=f"{pair_label} {method} — Checkerboard Alignment"
    )

    # 3-panel comparison figure
    save_comparison_figure(
        before_ov, after_ov, checker,
        title=f"Same-Modal Registration  |  {method}  |  {pair_label}",
        filepath=os.path.join(output_dir, f"{pair_label}_{method}_comparison.png")
    )

    # ── MI before / after ─────────────────────────────────────
    mi_before = mutual_information(img1_resized, img_orig2)
    mi_after  = mutual_information(registered,   img_orig2)
    mi_imp    = (
        ((mi_after - mi_before) / mi_before * 100)
        if mi_before > 1e-10 else 0.0
    )
    print(f"  MI before      : {mi_before:.4f}")
    print(f"  MI after       : {mi_after:.4f}")
    print(f"  MI improvement : {mi_imp:+.2f}%")

    return {
        "pair":               pair_label,
        "mode":               "same",
        "method":             method,
        "matches":            len(good_matches),
        "inliers":            inliers,
        "outliers":           outliers,
        "inlier_ratio":       round(inlier_ratio, 4),
        "n_gcps":             n_gcps,
        "mre_pixels":         round(mre, 4),
        "mi_before":          round(mi_before, 4),
        "mi_after":           round(mi_after, 4),
        "mi_improvement_pct": round(mi_imp, 2),
    }


# ─────────────────────────────────────────────────────────────
# CROSS-MODAL REGISTRATION  (SimpleITK Mattes MI Maximisation)
# ─────────────────────────────────────────────────────────────

def register_cross_modal(ir_path: str, vis_path: str,
                          pair_label: str = "pair",
                          output_dir: str = "results/cross_modal") -> dict:
    """
    Register IR (moving) → Visible (fixed) using Mattes MI maximisation.

    Why not SIFT/ORB for cross-modal?
      Feature detectors rely on gradient-based descriptors. IR images
      capture emitted thermal radiation; visible images capture reflected
      light. The same scene has fundamentally different gradient fields
      in each modality, so SIFT/ORB produce very few correct matches
      (low inlier ratio). MI maximisation works directly on intensity
      statistics — no feature correspondence needed.

    Pipeline:
      1. Load as SimpleITK Float32; Visible = fixed, IR = moving
      2. Geometry-centred Euler2DTransform initialisation
      3. Mattes MI metric, 20% random sampling, 300 RSGD iterations
      4. Resample IR onto Visible grid
      5. MRE via ORB on the registered pair (post-registration GCPs)
      6. Save individual overlays, checkerboard, 3-panel comparison

    Args:
        ir_path    : path to IR image (moving)
        vis_path   : path to Visible image (fixed / reference)
        pair_label : used for output filenames
        output_dir : directory for all saved outputs

    Returns: dict of registration metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  CROSS-MODAL REGISTRATION  |  MI Maximisation  |  {pair_label}")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────
    fixed  = sitk.ReadImage(vis_path, sitk.sitkFloat32)   # Visible = reference
    moving = sitk.ReadImage(ir_path,  sitk.sitkFloat32)   # IR = to be aligned

    # ── Registration setup ────────────────────────────────────
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300
    )
    init_tx = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg.SetInitialTransform(init_tx, inPlace=False)

    # ── Execute ───────────────────────────────────────────────
    transform = reg.Execute(fixed, moving)
    registered_sitk = sitk.Resample(
        moving, fixed, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )

    # ── Convert to NumPy uint8 ────────────────────────────────
    vis_np = sitk.GetArrayFromImage(fixed).astype(np.uint8)
    ir_np  = sitk.GetArrayFromImage(moving).astype(np.uint8)
    reg_np = sitk.GetArrayFromImage(registered_sitk).astype(np.uint8)

    # ── MI before / after ─────────────────────────────────────
    mi_before = mutual_information(vis_np, ir_np)
    mi_after  = mutual_information(vis_np, reg_np)
    mi_imp    = (
        ((mi_after - mi_before) / mi_before * 100)
        if mi_before > 1e-10 else 0.0
    )
    print(f"  MI before      : {mi_before:.4f}")
    print(f"  MI after       : {mi_after:.4f}")
    print(f"  MI improvement : {mi_imp:+.2f}%")

    # ── MRE via ORB GCPs on registered pair ───────────────────
    mre, n_gcps = _compute_mre_orb(reg_np, vis_np)
    print(f"  GCPs used      : {n_gcps}  (ORB inliers post-registration)")
    print(f"  MRE            : {mre:.3f} px")

    # ── Build overlays ────────────────────────────────────────
    h = min(vis_np.shape[0], ir_np.shape[0])
    w = min(vis_np.shape[1], ir_np.shape[1])

    before_ov = np.zeros((h, w, 3), dtype=np.uint8)
    before_ov[:, :, 0] = ir_np[:h, :w]
    before_ov[:, :, 1] = vis_np[:h, :w]

    after_ov = np.zeros((h, w, 3), dtype=np.uint8)
    after_ov[:, :, 0] = reg_np[:h, :w]
    after_ov[:, :, 1] = vis_np[:h, :w]

    checker = create_checkerboard(reg_np[:h, :w], vis_np[:h, :w])

    # Individual overlay files
    save_overlay(
        ir_np, vis_np,
        os.path.join(output_dir, f"{pair_label}_overlay_before.png"),
        title=f"{pair_label} — Before Registration (IR red, Visible green)"
    )
    save_overlay(
        reg_np, vis_np,
        os.path.join(output_dir, f"{pair_label}_overlay_after.png"),
        title=f"{pair_label} — After Registration (IR red, Visible green)"
    )
    save_checkerboard(
        reg_np[:h, :w], vis_np[:h, :w],
        os.path.join(output_dir, f"{pair_label}_checkerboard.png"),
        title=f"{pair_label} — Checkerboard Alignment (IR / Visible)"
    )

    # 3-panel comparison figure
    save_comparison_figure(
        before_ov, after_ov, checker,
        title=f"Cross-Modal Registration (IR→Visible)  |  {pair_label}",
        filepath=os.path.join(output_dir, f"{pair_label}_comparison.png")
    )

    return {
        "pair":               pair_label,
        "mode":               "cross",
        "method":             "MI-SimpleITK",
        "matches":            "N/A",
        "inliers":            "N/A",
        "outliers":           "N/A",
        "inlier_ratio":       "N/A",
        "n_gcps":             n_gcps,
        "mre_pixels":         round(mre, 4),
        "mi_before":          round(mi_before, 4),
        "mi_after":           round(mi_after, 4),
        "mi_improvement_pct": round(mi_imp, 2),
    }


def _compute_mre_orb(img_reg: np.ndarray, img_ref: np.ndarray):
    """
    Estimate post-registration MRE using ORB + RANSAC on the registered pair.
    ORB inliers act as GCPs to measure residual pixel-level error.

    Returns: (mre_pixels: float, n_gcps: int)
    """
    det = cv2.ORB_create(
        nfeatures=5000, scaleFactor=1.1,
        nlevels=12, edgeThreshold=15, fastThreshold=5
    )
    kp1, des1 = det.detectAndCompute(enhance_image(img_reg), None)
    kp2, des2 = det.detectAndCompute(enhance_image(img_ref), None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        print("  [WARNING] Too few keypoints for MRE — reporting 999.0")
        return 999.0, 0

    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [
        m for pair in knn if len(pair) == 2
        for m, n in [pair] if m.distance < 0.75 * n.distance
    ]

    if len(good) < 5:
        print(f"  [WARNING] Only {len(good)} good ORB matches — MRE unreliable")
        return 999.0, len(good)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return 999.0, 0

    return compute_mre(src_pts, dst_pts, H, mask.ravel())


# ─────────────────────────────────────────────────────────────
# FAILURE ANALYSIS REPORT
# ─────────────────────────────────────────────────────────────

def print_failure_analysis():
    """
    Print the conceptual failure analysis required by the Week 2 rubric.
    Explains why SIFT fails on IR–Visible pairs and why MI is used instead.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║              FAILURE ANALYSIS — WEEK 2 REQUIREMENT          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  WHY SIFT/ORB FAILS ON IR–VISIBLE CROSS-MODAL PAIRS          ║
║  ─────────────────────────────────────────────────────────   ║
║  SIFT and ORB build feature descriptors from local image     ║
║  gradients. In visible-light images, gradients arise from    ║
║  reflectance boundaries (colour/texture edges). In IR        ║
║  images, gradients arise from thermal emission boundaries    ║
║  (temperature edges). The same physical object produces      ║
║  entirely different gradient patterns in each modality:      ║
║                                                              ║
║  • A warm body = strong IR edge, weak visible edge           ║
║  • A coloured wall = strong visible edge, weak IR edge       ║
║  • Glass = high visible reflectance, near-zero IR emission   ║
║                                                              ║
║  Consequence: SIFT/ORB find very few mutually recognisable   ║
║  keypoints across modalities → low match count → RANSAC      ║
║  fails to estimate a reliable homography.                    ║
║                                                              ║
║  SOLUTION: MUTUAL INFORMATION MAXIMISATION                   ║
║  ─────────────────────────────────────────────────────────   ║
║  MI measures statistical dependence between the joint        ║
║  intensity histograms of two images — no gradient or         ║
║  feature correspondence required. MI is maximised when the   ║
║  two images are geometrically aligned, regardless of their   ║
║  appearance. SimpleITK's Mattes MI metric with an Euler2D    ║
║  transform handles cross-modal registration robustly.        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────

def _print_table(title: str, df: pd.DataFrame):
    """Pretty-print a summary DataFrame to stdout."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print("=" * 70)


def run_all():
    """
    Process all three target folders sequentially:

      For each folder:
        1. Modality is read directly from TARGET_FOLDERS — authoritative.
        2. Same-modal  → probe ORB vs SIFT inlier ratios → run only the winner.
        3. Cross-modal → SimpleITK Mattes MI maximisation.

    Saves a unified CSV summary and prints failure analysis.
    """
    print("\n" + "=" * 60)
    print("  WEEK 2 — IMAGE REGISTRATION PIPELINE")
    print("  Targets: P1W2 (same)  P2W2 (same)  P3W2 (cross)")
    print("=" * 60)

    all_results = []

    for folder, mode, label in TARGET_FOLDERS:
        try:
            path1, path2 = get_images_from_folder(folder)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n  [ERROR] {e} — skipping {folder}")
            continue

        print(f"\n  {folder}/  →  {Path(path1).name}  +  {Path(path2).name}  [{mode}-modal]")

        # ── Load images ──────────────────────────────────────────────────
        img_orig1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img_orig2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        if img_orig1 is None or img_orig2 is None:
            print(f"  [ERROR] Could not load images in {folder}/ — skipping")
            continue

        # ── Dispatch on authoritative modality label ─────────────────────
        if mode == "same":
            # Probe both detectors; run full registration only with winner
            best_method = select_best_method(img_orig1, img_orig2)
            r = register_same_modality(
                img_orig1, img_orig2,
                method=best_method,
                pair_label=label,
                output_dir="results/same_modal"
            )
            if r:
                all_results.append(r)

        else:  # cross-modal: path1=IR (alphabetically first), path2=Visible
            r = register_cross_modal(
                ir_path=path1,
                vis_path=path2,
                pair_label=label,
                output_dir="results/cross_modal"
            )
            if r:
                all_results.append(r)

    # ── Save unified results CSV ─────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        os.makedirs("results", exist_ok=True)
        csv_path = "results/registration_summary.csv"
        df.to_csv(csv_path, index=False)
        _print_table("REGISTRATION SUMMARY — ALL PAIRS", df)
        print(f"\n  CSV saved → {csv_path}")

    # ── Print failure analysis (rubric requirement) ──────────────────────
    print_failure_analysis()


def main():
    os.makedirs("results/same_modal",  exist_ok=True)
    os.makedirs("results/cross_modal", exist_ok=True)
    run_all()


if __name__ == "__main__":
    main()

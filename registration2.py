"""
registration2.py
-----------------
Week 2 deliverable — Cross-modal IR-Visible registration.

Why this file exists alongside registration.py:
SIFT/ORB + RANSAC (registration.py) works for same-modality pairs because
both images share gradient/intensity statistics. IR and visible images of
the same scene do NOT share those statistics (a hot rock and a dark rock
look identical in IR but very different in visible light), so keypoint
descriptors rarely match across modalities. The standard fix is to register
using a similarity measure that doesn't assume similar intensities at
all -- Mutual Information (MI) -- optimized directly over a geometric
transform (here: affine), rather than via point correspondences.

This script:
  1. Runs SIFT feature matching + RANSAC anyway, on the cross-modal pair,
     and reports the match/inlier counts. This gives you the concrete
     numbers ("X good matches, Y% inliers") to cite as evidence in your
     Week 2 report for *why* SIFT fails on IR-visible pairs.
  2. Registers the pair using Mattes Mutual Information + an affine
     transform (SimpleITK), which does not require feature correspondence.
  3. Computes Mean Registration Error (MRE) in pixels from manually
     selected ground control points (GCPs).
  4. Produces the required figure: unregistered overlay | registered
     overlay | checkerboard.
  5. Can run on a single pair or batch over tno2/1 .. tno2/5.

Usage
-----
Single pair:
    python registration2.py --vis tno2/1/visible.bmp --ir tno2/1/ir.bmp \
        --out week2results

Batch over all numbered subfolders in tno2/:
    python registration2.py --batch tno2 --out week2results

Skip interactive GCP clicking (no MRE computed, useful for a quick run):
    python registration2.py --batch tno2 --out week2results --no-interactive

Requirements:
    pip install opencv-python numpy matplotlib SimpleITK
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def load_image_gray(path):
    """Load an image as float32 grayscale. Raises if the file can't be read."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype(np.float32)


def find_pair_in_folder(folder):
    """
    Locate the IR and visible image inside a tno2/<N> folder.

    Tries to detect by filename keywords first (ir/lwir/therm vs
    vis/rgb/eo/color). Falls back to alphabetical order with a warning if
    no keyword match is found -- TNO subsets aren't named consistently,
    so double-check the printed assignment for each pair.
    """
    folder = Path(folder)
    exts = ("*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files = sorted({f for ext in exts for f in folder.glob(ext)})
    if len(files) < 2:
        raise FileNotFoundError(f"Need 2 images in {folder}, found {len(files)}")

    def is_ir(name):
        n = name.lower()
        return any(k in n for k in ("ir", "lwir", "therm", "fir"))

    def is_vis(name):
        n = name.lower()
        return any(k in n for k in ("vis", "rgb", "eo", "color"))

    ir_candidates = [f for f in files if is_ir(f.name)]
    vis_candidates = [f for f in files if is_vis(f.name)]

    if ir_candidates and vis_candidates:
        return vis_candidates[0], ir_candidates[0]  # (fixed, moving)

    print(f"  [!] Could not auto-detect IR/visible by filename in {folder}; "
          f"assuming fixed={files[1].name}, moving={files[0].name}. "
          f"Check this is correct for this pair.")
    return files[1], files[0]


# --------------------------------------------------------------------------
# Step 1: demonstrate SIFT failure on cross-modal pairs
# --------------------------------------------------------------------------

def demo_feature_matching_failure(fixed_np, moving_np, out_path=None):
    """
    Run SIFT + BFMatcher + RANSAC homography on a cross-modal pair.
    Returns keypoint/match/inlier counts -- use these numbers directly
    in the report to justify switching to MI-based registration.
    """
    fixed_u8 = cv2.normalize(fixed_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    moving_u8 = cv2.normalize(moving_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(fixed_u8, None)
    kp2, des2 = sift.detectAndCompute(moving_u8, None)

    result = {
        "kp_fixed": len(kp1) if kp1 else 0,
        "kp_moving": len(kp2) if kp2 else 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "homography_found": False,
    }

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return result

    bf = cv2.BFMatcher()
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    result["good_matches"] = len(good)

    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None and mask is not None:
            inliers = int(mask.sum())
            result["inliers"] = inliers
            result["inlier_ratio"] = inliers / len(good)
            result["homography_found"] = True

        if out_path is not None:
            match_img = cv2.drawMatches(
                fixed_u8, kp1, moving_u8, kp2, good[:40], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(str(out_path), match_img)

    return result


# --------------------------------------------------------------------------
# Step 2: MI-based affine registration
# --------------------------------------------------------------------------

def register_mi_sitk(fixed_np, moving_np, num_histogram_bins=50,
                      sampling_pct=0.20, iterations=300, verbose=False):
    """
    Register moving_np onto fixed_np using an affine transform optimized
    against Mattes Mutual Information. Multi-resolution (4x -> 2x -> 1x)
    for robustness against local optima.

    Returns (final_transform, final_metric_value, stop_condition_string).
    final_transform maps a point in the FIXED image's coordinate space to
    the corresponding point in the MOVING image's coordinate space.
    """
    fixed = sitk.GetImageFromArray(fixed_np.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_np.astype(np.float32))

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(fixed.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=num_histogram_bins)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_pct)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=iterations,
        gradientMagnitudeTolerance=1e-8)
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(initial_transform, inPlace=False)

    if verbose:
        reg.AddCommand(
            sitk.sitkIterationEvent,
            lambda: print(f"    iter={reg.GetOptimizerIteration():3d}  "
                          f"MI={reg.GetMetricValue():.5f}"))

    final_transform = reg.Execute(fixed, moving)
    return final_transform, reg.GetMetricValue(), reg.GetOptimizerStopConditionDescription()


def resample_moving(fixed_np, moving_np, transform):
    """Warp the moving image into the fixed image's grid using the final transform."""
    fixed = sitk.GetImageFromArray(fixed_np.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_np.astype(np.float32))
    resampled = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0,
                               moving.GetPixelID())
    return sitk.GetArrayFromImage(resampled)


# --------------------------------------------------------------------------
# Step 3: ground control points + MRE
# --------------------------------------------------------------------------

def pick_gcps(img, n_points=5, title="Click points"):
    """Interactively click n_points on img; returns an (n_points, 2) array of (x, y)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, cmap="gray")
    ax.set_title(f"{title}\nClick {n_points} points in order, window closes automatically")
    pts = plt.ginput(n_points, timeout=0)
    plt.close(fig)
    return np.array(pts)


def save_gcps(path, fixed_pts, moving_pts):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fx", "fy", "mx", "my"])
        for (fx, fy), (mx, my) in zip(fixed_pts, moving_pts):
            writer.writerow([fx, fy, mx, my])


def load_gcps(path):
    fixed_pts, moving_pts = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            fx, fy, mx, my = map(float, row)
            fixed_pts.append((fx, fy))
            moving_pts.append((mx, my))
    return np.array(fixed_pts), np.array(moving_pts)


def compute_mre(transform, fixed_pts, moving_pts):
    """
    Mean Registration Error: for each GCP pair, map the fixed-image point
    through the learned transform and compare to where the corresponding
    point actually is in the moving image.
    """
    errors = []
    for (fx, fy), (mx, my) in zip(fixed_pts, moving_pts):
        pred_x, pred_y = transform.TransformPoint((float(fx), float(fy)))
        errors.append(np.hypot(pred_x - mx, pred_y - my))
    errors = np.array(errors)
    return errors.mean(), errors.std(), errors


# --------------------------------------------------------------------------
# Step 4: visualizations
# --------------------------------------------------------------------------

def make_checkerboard(img1, img2, tile=24):
    """Alternate tile-sized blocks from img1/img2 -- misalignment shows as
    discontinuities at tile boundaries."""
    h, w = img1.shape[:2]
    board = np.zeros((h, w), dtype=img1.dtype)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            use_img1 = ((x // tile) + (y // tile)) % 2 == 0
            src = img1 if use_img1 else img2
            board[y:y + tile, x:x + tile] = src[y:y + tile, x:x + tile]
    return board


def make_color_overlay(img1, img2):
    """False-color overlay: img1 in red channel, img2 in green channel."""
    def norm(im):
        im = im.astype(np.float32)
        rng = im.max() - im.min()
        return (im - im.min()) / rng if rng > 0 else np.zeros_like(im)

    r, g = norm(img1), norm(img2)
    b = np.zeros_like(r)
    return np.dstack([r, g, b])


def plot_comparison(fixed, moving_unreg_display, moving_registered, pair_name, out_dir):
    """Required figure: unregistered overlay | registered overlay | checkerboard."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(make_color_overlay(fixed, moving_unreg_display))
    axes[0].set_title("Unregistered overlay\n(red=visible, green=IR)")
    axes[0].set_xlabel("x (pixels)")
    axes[0].set_ylabel("y (pixels)")

    axes[1].imshow(make_color_overlay(fixed, moving_registered))
    axes[1].set_title("Registered overlay\n(red=visible, green=IR)")
    axes[1].set_xlabel("x (pixels)")

    cb = make_checkerboard(fixed, moving_registered)
    axes[2].imshow(cb, cmap="gray")
    axes[2].set_title("Checkerboard\n(post-registration)")
    axes[2].set_xlabel("x (pixels)")

    fig.suptitle(f"Registration result: {pair_name}")
    fig.tight_layout()
    out_path = Path(out_dir) / f"{pair_name}_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def process_pair(fixed_path, moving_path, pair_name, out_dir,
                  gcp_file=None, n_gcp=5, interactive=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed = load_image_gray(fixed_path)
    moving = load_image_gray(moving_path)

    print(f"\n=== Pair: {pair_name} ===")
    print(f"Fixed (visible): {fixed_path}  shape={fixed.shape}")
    print(f"Moving (IR):     {moving_path}  shape={moving.shape}")

    # Step 1 -- SIFT failure demo (gives RANSAC inlier/outlier numbers for the report)
    match_png = out_dir / f"{pair_name}_sift_matches.png"
    sift_result = demo_feature_matching_failure(fixed, moving, out_path=match_png)
    print(f"SIFT cross-modal matching: {sift_result}")

    # Step 2 -- MI-based registration
    print("Running MI-based registration (SimpleITK)...")
    transform, final_mi, stop_cond = register_mi_sitk(fixed, moving)
    print(f"Final MI value: {final_mi:.5f} | stop condition: {stop_cond}")

    moving_registered = resample_moving(fixed, moving, transform)

    # Step 3 -- GCPs / MRE
    gcp_path = out_dir / f"{pair_name}_gcps.csv"
    fixed_pts, moving_pts = None, None
    if gcp_file and Path(gcp_file).exists():
        fixed_pts, moving_pts = load_gcps(gcp_file)
    elif interactive:
        print(f"Click {n_gcp} points on the VISIBLE (fixed) image...")
        fixed_pts = pick_gcps(fixed, n_gcp, title=f"{pair_name}: visible (fixed)")
        print(f"Now click the SAME {n_gcp} points, SAME order, on the IR (moving) image...")
        moving_pts = pick_gcps(moving, n_gcp, title=f"{pair_name}: IR (moving)")
        save_gcps(gcp_path, fixed_pts, moving_pts)
    else:
        print("No GCPs provided and interactive mode off -- MRE not computed.")

    mre_mean = mre_std = None
    if fixed_pts is not None and len(fixed_pts) > 0:
        mre_mean, mre_std, errors = compute_mre(transform, fixed_pts, moving_pts)
        print(f"MRE = {mre_mean:.2f} +/- {mre_std:.2f} px over {len(errors)} GCPs")

    # Step 4 -- figure
    if moving.shape != fixed.shape:
        moving_display = cv2.resize(moving, (fixed.shape[1], fixed.shape[0]))
    else:
        moving_display = moving
    fig_path = plot_comparison(fixed, moving_display, moving_registered, pair_name, out_dir)

    return {
        "pair": pair_name,
        "sift_good_matches": sift_result["good_matches"],
        "sift_inliers": sift_result["inliers"],
        "sift_inlier_ratio": round(sift_result["inlier_ratio"], 3),
        "mi_final_value": round(final_mi, 5),
        "mre_px": round(mre_mean, 3) if mre_mean is not None else "",
        "mre_std_px": round(mre_std, 3) if mre_std is not None else "",
        "n_gcps": len(fixed_pts) if fixed_pts is not None else 0,
        "figure": str(fig_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Week 2: cross-modal IR-visible registration via Mutual Information")
    parser.add_argument("--ir", type=str, help="Path to IR (moving) image")
    parser.add_argument("--vis", type=str, help="Path to visible (fixed) image")
    parser.add_argument("--batch", type=str,
                         help="Folder with numbered subfolders (e.g. tno2), each "
                              "containing one IR/visible pair")
    parser.add_argument("--out", type=str, default="week2results", help="Output directory")
    parser.add_argument("--n-gcp", type=int, default=5, help="GCPs per pair (min 5 required)")
    parser.add_argument("--no-interactive", action="store_true",
                         help="Skip interactive GCP picking (no MRE computed)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.batch:
        batch_root = Path(args.batch)
        subdirs = sorted([d for d in batch_root.iterdir() if d.is_dir()], key=lambda d: d.name)
        if not subdirs:
            raise FileNotFoundError(f"No subfolders found under {batch_root}")
        for d in subdirs:
            try:
                vis_path, ir_path = find_pair_in_folder(d)
                row = process_pair(
                    vis_path, ir_path, pair_name=d.name, out_dir=out_dir,
                    n_gcp=args.n_gcp, interactive=not args.no_interactive)
                rows.append(row)
            except Exception as e:
                print(f"  [!] Skipping {d.name}: {e}")
    elif args.ir and args.vis:
        row = process_pair(
            args.vis, args.ir, pair_name=Path(args.vis).parent.name or Path(args.vis).stem,
            out_dir=out_dir, n_gcp=args.n_gcp, interactive=not args.no_interactive)
        rows.append(row)
    else:
        parser.error("Provide either --batch <folder> or both --ir and --vis")

    csv_path = out_dir / "registration_error_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair", "sift_good_matches", "sift_inliers", "sift_inlier_ratio",
            "mi_final_value", "mre_px", "mre_std_px", "n_gcps", "figure"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved error table to {csv_path}")


if __name__ == "__main__":
    main()
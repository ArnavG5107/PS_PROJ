import cv2
import numpy as np
import os
from sklearn.metrics import mutual_info_score


# =====================================================
# MUTUAL INFORMATION
# =====================================================

def mutual_information(img1, img2):

    hist_2d, _, _ = np.histogram2d(
        img1.ravel(),
        img2.ravel(),
        bins=256
    )

    return mutual_info_score(
        None,
        None,
        contingency=hist_2d
    )


# =====================================================
# REPROJECTION ERROR
# =====================================================

def reprojection_error(
    src_pts,
    dst_pts,
    H,
    mask
):

    projected = cv2.perspectiveTransform(
        src_pts,
        H
    )

    mask = mask.ravel().astype(bool)

    error = np.linalg.norm(
        projected[mask] - dst_pts[mask],
        axis=2
    )

    return np.mean(error)


# =====================================================
# IMAGE ENHANCEMENT
# =====================================================

def enhance_image(img):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(img)


# =====================================================
# REGISTRATION PIPELINE
# =====================================================

def register_images(
    img1,
    img2,
    method="ORB"
):

    print("\n" + "=" * 60)
    print(method)
    print("=" * 60)

    if method == "ORB":

        detector = cv2.ORB_create(
            nfeatures=10000,
            scaleFactor=1.1,
            nlevels=12,
            edgeThreshold=15,
            fastThreshold=5
        )

        norm_type = cv2.NORM_HAMMING

    else:

        detector = cv2.SIFT_create(
            nfeatures=5000,
            contrastThreshold=0.01,
            edgeThreshold=5
        )

        norm_type = cv2.NORM_L2

    kp1, des1 = detector.detectAndCompute(
        img1,
        None
    )

    kp2, des2 = detector.detectAndCompute(
        img2,
        None
    )

    print(
        f"Keypoints Image A : {len(kp1)}"
    )

    print(
        f"Keypoints Image B : {len(kp2)}"
    )

    # -------------------------------------------------
    # SAVE KEYPOINTS
    # -------------------------------------------------

    kp_img1 = cv2.drawKeypoints(
        img1,
        kp1,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    kp_img2 = cv2.drawKeypoints(
        img2,
        kp2,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(
        f"results/{method}_keypoints_A.png",
        kp_img1
    )

    cv2.imwrite(
        f"results/{method}_keypoints_B.png",
        kp_img2
    )

    # -------------------------------------------------
    # MATCHING
    # -------------------------------------------------

    bf = cv2.BFMatcher(norm_type)

    knn_matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for pair in knn_matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:

            good_matches.append(m)

    print(
        f"Good Matches : {len(good_matches)}"
    )

    if len(good_matches) < 4:

        print(
            "Not enough matches for homography."
        )

        return None

    # -------------------------------------------------
    # SAVE MATCHES
    # -------------------------------------------------

    match_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches[:100],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        f"results/{method}_matches.png",
        match_img
    )

    # -------------------------------------------------
    # HOMOGRAPHY + RANSAC
    # -------------------------------------------------

    src_pts = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if H is None:

        print(
            "Homography estimation failed."
        )

        return None

    mask = mask.ravel()

    inliers = np.sum(mask)
    outliers = len(mask) - inliers

    print(
        f"Inliers : {inliers}"
    )

    print(
        f"Outliers : {outliers}"
    )

    print(
        f"Inlier Ratio : "
        f"{inliers/len(mask):.3f}"
    )

    # -------------------------------------------------
    # SAVE INLIERS
    # -------------------------------------------------

    inlier_matches = [
        good_matches[i]
        for i in range(len(good_matches))
        if mask[i]
    ]

    inlier_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        f"results/{method}_inliers.png",
        inlier_img
    )

    # -------------------------------------------------
    # REPROJECTION ERROR
    # -------------------------------------------------

    rep_error = reprojection_error(
        src_pts,
        dst_pts,
        H,
        mask
    )

    print(
        f"Mean Reprojection Error : "
        f"{rep_error:.3f}"
    )

    # -------------------------------------------------
    # REGISTRATION
    # -------------------------------------------------

    registered = cv2.warpPerspective(
        img1,
        H,
        (
            img2.shape[1],
            img2.shape[0]
        )
    )

    cv2.imwrite(
        f"results/{method}_registered.png",
        registered
    )

    # -------------------------------------------------
    # OVERLAYS
    # -------------------------------------------------

    img1_resized = cv2.resize(
        img1,
        (
            img2.shape[1],
            img2.shape[0]
        )
    )

    before_overlay = cv2.addWeighted(
        img1_resized,
        0.5,
        img2,
        0.5,
        0
    )

    after_overlay = cv2.addWeighted(
        registered,
        0.5,
        img2,
        0.5,
        0
    )

    cv2.imwrite(
        f"results/{method}_overlay_before.png",
        before_overlay
    )

    cv2.imwrite(
        f"results/{method}_overlay_after.png",
        after_overlay
    )

    # -------------------------------------------------
    # MUTUAL INFORMATION
    # -------------------------------------------------

    mi_before = mutual_information(
        img1_resized,
        img2
    )

    mi_after = mutual_information(
        registered,
        img2
    )

    print(
        f"MI Before : {mi_before:.4f}"
    )

    print(
        f"MI After  : {mi_after:.4f}"
    )

    if mi_before > 1e-10:

        improvement = (
            (mi_after - mi_before)
            / mi_before
        ) * 100

        print(
            f"MI Improvement : "
            f"{improvement:.2f}%"
        )

    return {
        "method": method,
        "matches": len(good_matches),
        "inliers": int(inliers),
        "mi_after": mi_after,
        "error": rep_error
    }


# =====================================================
# MAIN
# =====================================================

def main():

    img1 = cv2.imread(
        "same/A.bmp",
        cv2.IMREAD_GRAYSCALE
    )

    img2 = cv2.imread(
        "same/B.bmp",
        cv2.IMREAD_GRAYSCALE
    )

    if img1 is None or img2 is None:

        print(
            "Images not found."
        )

        return

    print(
        f"A shape : {img1.shape}"
    )

    print(
        f"B shape : {img2.shape}"
    )

    img1 = enhance_image(img1)
    img2 = enhance_image(img2)

    os.makedirs(
        "results",
        exist_ok=True
    )

    orb = register_images(
        img1,
        img2,
        "ORB"
    )

    sift = register_images(
        img1,
        img2,
        "SIFT"
    )

    print("\n")
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    for result in [orb, sift]:

        if result is None:
            continue

        print(
            f"\n{result['method']}"
        )

        print(
            f"Matches : {result['matches']}"
        )

        print(
            f"Inliers : {result['inliers']}"
        )

        print(
            f"MI After : {result['mi_after']:.4f}"
        )

        print(
            f"Reprojection Error : "
            f"{result['error']:.3f}"
        )


if __name__ == "__main__":
    main()
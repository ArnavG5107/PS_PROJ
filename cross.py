import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt


DATASET_DIR = "imagesw2"

PAIRS = [
    ("IR1.bmp", "V1.bmp"),
    ("IR2.bmp", "V2.bmp"),
    ("IR3.bmp", "V3.bmp"),
    ("IR4.bmp", "V4.bmp"),
    ("IR5.bmp", "V5.bmp")
]


def mutual_information(img1, img2):

    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    mi = mutual_info_score(
        img1.flatten(),
        img2.flatten()
    )

    return mi


def save_overlay(img1, img2, filename):

    overlay = np.zeros(
        (img1.shape[0], img1.shape[1], 3),
        dtype=np.uint8
    )

    overlay[:, :, 0] = img1
    overlay[:, :, 1] = img2

    plt.figure(figsize=(8, 6))

    plt.imshow(overlay)

    plt.axis("off")

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def create_checkerboard(img1, img2, block_size=32):

    h, w = img1.shape

    checker = np.zeros_like(img1)

    for y in range(0, h, block_size):

        for x in range(0, w, block_size):

            if ((x // block_size) +
                (y // block_size)) % 2 == 0:

                checker[
                    y:y+block_size,
                    x:x+block_size
                ] = img1[
                    y:y+block_size,
                    x:x+block_size
                ]

            else:

                checker[
                    y:y+block_size,
                    x:x+block_size
                ] = img2[
                    y:y+block_size,
                    x:x+block_size
                ]

    return checker


def save_checkerboard(img1, img2, filename):

    checker = create_checkerboard(
        img1,
        img2
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(
        checker,
        cmap="gray"
    )

    plt.axis("off")

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def register_mi(ir_path, vis_path):

    fixed = sitk.ReadImage(
        vis_path,
        sitk.sitkFloat32
    )

    moving = sitk.ReadImage(
        ir_path,
        sitk.sitkFloat32
    )

    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50
    )

    registration.SetMetricSamplingStrategy(
        registration.RANDOM
    )

    registration.SetMetricSamplingPercentage(
        0.20
    )

    registration.SetInterpolator(
        sitk.sitkLinear
    )

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=300
    )

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration.SetInitialTransform(
        initial_transform,
        inPlace=False
    )

    transform = registration.Execute(
        fixed,
        moving
    )

    registered = sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID()
    )

    return fixed, moving, registered


def main():

    os.makedirs(
        "results/cross_modal",
        exist_ok=True
    )

    summary = []

    print("\n=== MI REGISTRATION ===\n")

    for idx, (ir_name, vis_name) in enumerate(PAIRS):

        ir_path = os.path.join(
            DATASET_DIR,
            ir_name
        )

        vis_path = os.path.join(
            DATASET_DIR,
            vis_name
        )

        fixed, moving, registered = register_mi(
            ir_path,
            vis_path
        )

        vis_np = sitk.GetArrayFromImage(
            fixed
        ).astype(np.uint8)

        ir_np = sitk.GetArrayFromImage(
            moving
        ).astype(np.uint8)

        reg_np = sitk.GetArrayFromImage(
            registered
        ).astype(np.uint8)

        mi_before = mutual_information(
            vis_np,
            ir_np
        )

        mi_after = mutual_information(
            vis_np,
            reg_np
        )

        save_overlay(
            ir_np,
            vis_np,
            f"results/cross_modal/pair{idx+1}_before.png"
        )

        save_overlay(
            reg_np,
            vis_np,
            f"results/cross_modal/pair{idx+1}_after.png"
        )

        save_checkerboard(
            reg_np,
            vis_np,
            f"results/cross_modal/pair{idx+1}_checkerboard.png"
        )

        summary.append([
            idx + 1,
            round(mi_before, 4),
            round(mi_after, 4)
        ])

        print(
            f"Pair {idx+1}"
        )

        print(
            f"MI Before : {mi_before:.4f}"
        )

        print(
            f"MI After  : {mi_after:.4f}"
        )

        print("-" * 40)

    df = pd.DataFrame(
        summary,
        columns=[
            "Pair",
            "MI_Before",
            "MI_After"
        ]
    )

    df.to_csv(
        "results/cross_modal/mi_results.csv",
        index=False
    )

    print("\nResults saved in:")
    print("results/cross_modal/")


if __name__ == "__main__":
    main()
from PIL import Image
import numpy as np
from sklearn.metrics import mutual_info_score


def load_grayscale_image(path):
    """
    Load image and convert to grayscale.
    """
    img = Image.open(path).convert("L")
    return np.array(img)


def compute_mutual_information(img1, img2):
    """
    Compute Mutual Information between two images.
    """

    # Resize by cropping to common size
    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])

    img1 = img1[:min_h, :min_w]
    img2 = img2[:min_h, :min_w]

    mi = mutual_info_score(
        img1.flatten(),
        img2.flatten()
    )

    return mi


def main():

    visible = load_grayscale_image(
        "images2/visible.jpg"
    )

    infrared = load_grayscale_image(
        "images2/infrared2.png"
    )

    mi = compute_mutual_information(
        visible,
        infrared
    )

    print("\n==============================")
    print("MUTUAL INFORMATION RESULT")
    print("==============================")
    print(f"Visible shape  : {visible.shape}")
    print(f"Infrared shape : {infrared.shape}")
    print(f"MI Value       : {mi:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    main()
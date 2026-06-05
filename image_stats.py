from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import os


def load_grayscale_image(image_path):
    """
    Load image and convert to grayscale.
    """
    img = Image.open(image_path).convert("L")
    return np.array(img)


def compute_mean(image):
    """
    Compute mean pixel intensity.
    """
    return np.mean(image)


def compute_variance(image):
    """
    Compute variance of pixel intensity.
    """
    return np.var(image)


def compute_histogram(image):
    """
    Compute histogram with 256 bins.
    """
    hist, bins = np.histogram(
        image.flatten(),
        bins=256,
        range=(0, 256)
    )

    return hist, bins


def compute_entropy(image):
    """
    Compute Shannon entropy.
    """
    hist, _ = compute_histogram(image)

    prob = hist / np.sum(hist)

    prob = prob[prob > 0]

    entropy = -np.sum(
        prob * np.log2(prob)
    )

    return entropy


def compute_mutual_information(image1, image2):
    """
    Compute Mutual Information between two images.
    """

    min_h = min(
        image1.shape[0],
        image2.shape[0]
    )

    min_w = min(
        image1.shape[1],
        image2.shape[1]
    )

    image1 = image1[:min_h, :min_w]
    image2 = image2[:min_h, :min_w]

    mi = mutual_info_score(
        image1.flatten(),
        image2.flatten()
    )

    return mi


def save_histogram(image, image_name):
    """
    Save histogram plot.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        image.flatten(),
        bins=256,
        color="black"
    )

    plt.title(
        f"Histogram - {image_name}"
    )

    plt.xlabel(
        "Pixel Intensity"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        f"plots/{image_name}_histogram.png",
        dpi=300
    )

    plt.close()


def save_dft_spectrum(image, image_name):
    """
    Compute and save DFT magnitude spectrum.
    """

    fft = np.fft.fft2(image)

    fft_shift = np.fft.fftshift(fft)

    magnitude = np.log(
        np.abs(fft_shift) + 1
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(
        magnitude,
        cmap="gray"
    )

    plt.title(
        f"DFT Magnitude Spectrum - {image_name}"
    )

    plt.xlabel(
        "Frequency X"
    )

    plt.ylabel(
        "Frequency Y"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        f"plots/{image_name}_dft.png",
        dpi=300
    )

    plt.close()


def analyze_image(image_path):
    """
    Analyze one image.
    """

    image = load_grayscale_image(
        image_path
    )

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    mean = compute_mean(image)

    variance = compute_variance(image)

    entropy = compute_entropy(image)

    save_histogram(
        image,
        image_name
    )

    save_dft_spectrum(
        image,
        image_name
    )

    print("\n========================")
    print(f"Image : {image_name}")
    print("========================")

    print(
        "Shape    :",
        image.shape
    )

    print(
        "Mean     :",
        round(mean, 4)
    )

    print(
        "Variance :",
        round(variance, 4)
    )

    print(
        "Entropy  :",
        round(entropy, 4)
    )

    return image, image_name


def main():

    os.makedirs(
        "plots",
        exist_ok=True
    )

    image_files = [
        "images/animal.jpg",
        "images/city.jpg",
        "images/nature.jpg"
    ]

    loaded_images = []
    image_names = []

    print("\n")
    print("=" * 50)
    print("IMAGE STATISTICS")
    print("=" * 50)

    for image_path in image_files:

        image, name = analyze_image(
            image_path
        )

        loaded_images.append(
            image
        )

        image_names.append(
            name
        )

    print("\n")
    print("=" * 50)
    print("MUTUAL INFORMATION")
    print("=" * 50)

    for i in range(
        len(loaded_images)
    ):

        for j in range(
            i + 1,
            len(loaded_images)
        ):

            mi = compute_mutual_information(
                loaded_images[i],
                loaded_images[j]
            )

            print(
                f"MI ({image_names[i]} vs {image_names[j]})"
                f" = {mi:.4f}"
            )


if __name__ == "__main__":
    main()

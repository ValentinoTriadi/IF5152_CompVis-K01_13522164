# Nama: Valentino Chryslie Triadi
# NIM: 13522164
# Fitur unik: Filtering

# Import dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import data

# Load image example from skimage
camera = data.camera()
coin = data.coins()
checkerboard = data.checkerboard()
astronaut = data.astronaut()
chelsea = data.chelsea()

# Load image personal
personal_image_path = Path(__file__).parent / "personal_image.png"
personal_image = cv2.imread(str(personal_image_path))


class ImageFilter:
    """Class for various image filtering operations"""

    def __init__(self, image, image_name="image"):
        """
        Initialize with an image

        Args:
            image: Input image
            image_name: Name prefix for saved files
        """
        self.original = image
        self.image_name = image_name
        if self.original is None:
            raise ValueError(f"Cannot load image {image}")

    def gaussian_blur(self, kernel_size=(5, 5), sigma=1.0):
        """
        Apply Gaussian blur filter

        Args:
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma: Standard deviation of the Gaussian kernel

        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(self.original, kernel_size, sigma)

    def median_filter(self, kernel_size=5):
        """
        Apply median filter (good for salt-and-pepper noise)

        Args:
            kernel_size: Size of the kernel (must be odd)

        Returns:
            Filtered image
        """
        return cv2.medianBlur(self.original, kernel_size)

    def visualize_all_filters(self, output_dir=None, save_path=None):
        """
        Visualize all filtering results

        Args:
            save_path: Path to save the visualization
        """
        gaussian = self.gaussian_blur()
        gaussian_kernel_9 = self.gaussian_blur(kernel_size=(9, 9))
        gaussian_kernel_25 = self.gaussian_blur(kernel_size=(25, 25))
        gaussian_sigma_2 = self.gaussian_blur(sigma=2.0)
        gaussian_sigma_10 = self.gaussian_blur(sigma=10.0)
        median = self.median_filter()
        median_kernel_9 = self.median_filter(kernel_size=9)
        median_kernel_25 = self.median_filter(kernel_size=25)

        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(25, 25))

        images = [
            (self.original, self.image_name + "Original Image"),
            (gaussian, self.image_name + "Gaussian Blur (5x5 Kernel)"),
            (gaussian_kernel_25, self.image_name + "Gaussian Blur (25x25 Kernel)"),
            (gaussian_kernel_9, self.image_name + "Gaussian Blur (9x9 Kernel)"),
            (gaussian_sigma_2, self.image_name + "Gaussian Blur (Sigma=2.0)"),
            (gaussian_sigma_10, self.image_name + "Gaussian Blur (Sigma=10.0)"),
            (median, self.image_name + "Median Filter (5x5 Kernel)"),
            (median_kernel_9, self.image_name + "Median Filter (9x9 Kernel)"),
            (median_kernel_25, self.image_name + "Median Filter (25x25 Kernel)"),
        ]

        for ax, (img, title) in zip(axes.flat, images):
            if len(img.shape) == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()

        if output_dir and save_path:
            plt.savefig(output_dir / save_path, dpi=300, bbox_inches="tight")

            for i in images:
                cv2.imwrite(output_dir / f"{i[1].replace(' ', '_').lower()}.png", i[0])

            print(f"Visualization saved to {output_dir / save_path}")

        plt.close(fig)


def main():
    """Main function to demonstrate filtering"""
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Apply filters
    print("Applying image filters...")
    camera_filter = ImageFilter(camera, image_name="Camera ")
    coin_filter = ImageFilter(coin, image_name="Coin ")
    checkerboard_filter = ImageFilter(checkerboard, image_name="Checkerboard ")
    astronaut_filter = ImageFilter(astronaut, image_name="Astronaut ")
    chelsea_filter = ImageFilter(chelsea, image_name="Chelsea ")
    personal_image_filter = ImageFilter(personal_image, image_name="Personal_Image ")

    # Visualize results
    camera_filter.visualize_all_filters(
        output_dir=output_dir, save_path="camera_filtering_results.png"
    )
    coin_filter.visualize_all_filters(
        output_dir=output_dir, save_path="coin_filtering_results.png"
    )
    checkerboard_filter.visualize_all_filters(
        output_dir=output_dir, save_path="checkerboard_filtering_results.png"
    )
    astronaut_filter.visualize_all_filters(
        output_dir=output_dir, save_path="astronaut_filtering_results.png"
    )
    chelsea_filter.visualize_all_filters(
        output_dir=output_dir, save_path="chelsea_filtering_results.png"
    )
    personal_image_filter.visualize_all_filters(
        output_dir=output_dir, save_path="personal_image_filtering_results.png"
    )

    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()

# Nama: Valentino Chryslie Triadi
# NIM: 13522164
# Fitur unik: Edge Detection

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


class EdgeDetector:
    """Class for various edge detection operations"""

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

        # Handle grayscale images
        if len(self.original.shape) == 2:
            self.gray = self.original
        else:
            self.gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)

    def sobel_edge_detection(self, kernel_size=3):
        """
        Sobel edge detection (gradient-based)

        Args:
            kernel_size: Size of the Sobel kernel (1, 3, 5, or 7)

        Returns:
            Edge image
        """
        # Sobel in X direction
        sobel_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=kernel_size)

        # Sobel in Y direction
        sobel_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        # Combine gradients
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(sobel / sobel.max() * 255)

        return sobel

    def canny_edge_detection(self, threshold1=50, threshold2=150):
        """
        Canny edge detection (multi-stage algorithm)

        Args:
            threshold1: Lower threshold for hysteresis
            threshold2: Upper threshold for hysteresis

        Returns:
            Edge image
        """
        return cv2.Canny(self.gray, threshold1, threshold2)

    def visualize_all_edges(self, output_dir=None, save_path=None):
        """
        Visualize all edge detection results

        Args:
            output_dir: Directory to save the individual images
            save_path: Path to save the combined visualization
        """
        # Apply edge detection methods with different parameters
        sobel_3 = self.sobel_edge_detection(kernel_size=3)
        sobel_5 = self.sobel_edge_detection(kernel_size=5)
        sobel_15 = self.sobel_edge_detection(kernel_size=15)
        sobel_25 = self.sobel_edge_detection(kernel_size=25)
        canny_low = self.canny_edge_detection(threshold1=30, threshold2=100)
        canny_med = self.canny_edge_detection(threshold1=50, threshold2=150)
        canny_high = self.canny_edge_detection(threshold1=100, threshold2=200)
        canny_very_high = self.canny_edge_detection(threshold1=150, threshold2=250)

        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(25, 25))

        images = [
            (self.gray, self.image_name + "Original Image"),
            (sobel_3, self.image_name + "Sobel (3x3 Kernel)"),
            (sobel_5, self.image_name + "Sobel (5x5 Kernel)"),
            (sobel_15, self.image_name + "Sobel (15x15 Kernel)"),
            (sobel_25, self.image_name + "Sobel (25x25 Kernel)"),
            (canny_low, self.image_name + "Canny (Threshold 30-100)"),
            (canny_med, self.image_name + "Canny (Threshold 50-150)"),
            (canny_high, self.image_name + "Canny (Threshold 100-200)"),
            (canny_very_high, self.image_name + "Canny (Threshold 150-250)"),
        ]

        for ax, (img, title) in zip(axes.flat, images):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()

        if output_dir and save_path:
            plt.savefig(output_dir / save_path, dpi=300, bbox_inches="tight")

            for i in images:
                cv2.imwrite(
                    str(output_dir / f"{i[1].replace(' ', '_').lower()}.png"), i[0]
                )

            print(f"Visualization saved to {output_dir / save_path}")

        plt.close(fig)


def main():
    """Main function to demonstrate edge detection"""
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Apply edge detection
    print("Applying edge detection methods...")
    camera_detector = EdgeDetector(camera, image_name="Camera ")
    coin_detector = EdgeDetector(coin, image_name="Coin ")
    checkerboard_detector = EdgeDetector(checkerboard, image_name="Checkerboard ")
    astronaut_detector = EdgeDetector(astronaut, image_name="Astronaut ")
    chelsea_detector = EdgeDetector(chelsea, image_name="Chelsea ")
    personal_image_detector = EdgeDetector(personal_image, image_name="Personal_Image ")

    # Visualize results
    camera_detector.visualize_all_edges(
        output_dir=output_dir, save_path="camera_edge_detection_results.png"
    )
    coin_detector.visualize_all_edges(
        output_dir=output_dir, save_path="coin_edge_detection_results.png"
    )
    checkerboard_detector.visualize_all_edges(
        output_dir=output_dir, save_path="checkerboard_edge_detection_results.png"
    )
    astronaut_detector.visualize_all_edges(
        output_dir=output_dir, save_path="astronaut_edge_detection_results.png"
    )
    chelsea_detector.visualize_all_edges(
        output_dir=output_dir, save_path="chelsea_edge_detection_results.png"
    )
    personal_image_detector.visualize_all_edges(
        output_dir=output_dir, save_path="personal_image_edge_detection_results.png"
    )

    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()

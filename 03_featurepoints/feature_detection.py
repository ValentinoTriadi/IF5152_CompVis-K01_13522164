# Nama: Valentino Chryslie Triadi
# NIM: 13522164
# Fitur unik: Feature Point Detection

# Import dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import data
import json

# Load image example from skimage
camera = data.camera()
coin = data.coins()
checkerboard = data.checkerboard()
astronaut = data.astronaut()
chelsea = data.chelsea()

# Load image personal
personal_image_path = Path(__file__).parent / "personal_image.png"
personal_image = cv2.imread(str(personal_image_path))


class FeatureDetector:
    """Class for various feature point detection operations"""

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

        # Store statistics
        self.statistics = {}

    def harris_corner_detection(self, block_size=2, ksize=3, k=0.04):
        """
        Harris Corner Detection

        Args:
            block_size: Neighborhood size
            ksize: Aperture parameter for Sobel
            k: Harris detector free parameter

        Returns:
            Image with corners marked, number of corners, and statistics
        """
        # Detect corners
        gray_float = np.float32(self.gray)
        harris_response = cv2.cornerHarris(gray_float, block_size, ksize, k)

        # Dilate to mark corners
        harris_response = cv2.dilate(harris_response, None)

        # Create result image
        result = self.gray.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # Threshold for optimal corner detection
        threshold = 0.01 * harris_response.max()
        result[harris_response > threshold] = [0, 0, 255]

        # Count corners
        num_corners = np.sum(harris_response > threshold)

        # Calculate statistics
        stats = {
            "num_features": int(num_corners),
            "max_response": float(harris_response.max()),
            "mean_response": (
                float(harris_response[harris_response > threshold].mean())
                if num_corners > 0
                else 0
            ),
            "threshold": float(threshold),
        }

        return result, stats

    def fast_feature_detection(self, threshold=10):
        """
        FAST (Features from Accelerated Segment Test) Detection

        Args:
            threshold: Threshold for detection

        Returns:
            Image with keypoints marked and statistics
        """
        # Create FAST detector
        fast = cv2.FastFeatureDetector_create(threshold=threshold)

        # Detect keypoints
        keypoints = fast.detect(self.gray, None)

        # Draw keypoints
        result = cv2.drawKeypoints(
            (
                cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
                if len(self.gray.shape) == 2
                else self.gray
            ),
            keypoints,
            None,
            color=(255, 0, 0),
        )

        # Calculate statistics
        if keypoints:
            responses = [kp.response for kp in keypoints]
            sizes = [kp.size for kp in keypoints]
            stats = {
                "num_features": len(keypoints),
                "max_response": float(max(responses)),
                "mean_response": float(np.mean(responses)),
                "min_response": float(min(responses)),
                "mean_size": float(np.mean(sizes)),
            }
        else:
            stats = {
                "num_features": 0,
                "max_response": 0,
                "mean_response": 0,
                "min_response": 0,
                "mean_size": 0,
            }

        return result, stats

    def sift_feature_detection(self, n_features=0):
        """
        SIFT (Scale-Invariant Feature Transform) Detection

        Args:
            n_features: Number of best features to retain (0 = all)

        Returns:
            Image with keypoints marked and statistics
        """
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures=n_features)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(self.gray, None)

        # Draw keypoints
        result = cv2.drawKeypoints(
            (
                cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
                if len(self.gray.shape) == 2
                else self.gray
            ),
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        # Calculate statistics
        if keypoints:
            responses = [kp.response for kp in keypoints]
            sizes = [kp.size for kp in keypoints]
            stats = {
                "num_features": len(keypoints),
                "max_response": float(max(responses)),
                "mean_response": float(np.mean(responses)),
                "min_response": float(min(responses)),
                "mean_size": float(np.mean(sizes)),
                "max_size": float(max(sizes)),
                "min_size": float(min(sizes)),
            }
        else:
            stats = {
                "num_features": 0,
                "max_response": 0,
                "mean_response": 0,
                "min_response": 0,
                "mean_size": 0,
                "max_size": 0,
                "min_size": 0,
            }

        return result, stats

    def visualize_all_features(self, output_dir=None, save_path=None):
        """
        Visualize all feature detection results

        Args:
            output_dir: Directory to save the individual images
            save_path: Path to save the combined visualization
        """
        # Apply feature detection methods with different parameters
        harris_1, harris_stats_1 = self.harris_corner_detection(block_size=2, k=0.04)
        harris_2, harris_stats_2 = self.harris_corner_detection(block_size=3, k=0.04)
        harris_3, harris_stats_3 = self.harris_corner_detection(block_size=5, k=0.04)

        sift_1, sift_stats_1 = self.sift_feature_detection(n_features=100)
        sift_2, sift_stats_2 = self.sift_feature_detection(n_features=200)
        sift_3, sift_stats_3 = self.sift_feature_detection(n_features=0)

        fast_1, fast_stats_1 = self.fast_feature_detection(threshold=10)
        fast_2, fast_stats_2 = self.fast_feature_detection(threshold=25)
        fast_3, fast_stats_3 = self.fast_feature_detection(threshold=50)

        # Store all statistics
        all_stats = {
            f"{self.image_name}Harris (Block=2, k=0.04)": harris_stats_1,
            f"{self.image_name}Harris (Block=3, k=0.04)": harris_stats_2,
            f"{self.image_name}Harris (Block=5, k=0.04)": harris_stats_3,
            f"{self.image_name}SIFT (N=100)": sift_stats_1,
            f"{self.image_name}SIFT (N=200)": sift_stats_2,
            f"{self.image_name}SIFT (All)": sift_stats_3,
            f"{self.image_name}FAST (Threshold=10)": fast_stats_1,
            f"{self.image_name}FAST (Threshold=25)": fast_stats_2,
            f"{self.image_name}FAST (Threshold=50)": fast_stats_3,
        }

        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(25, 25))

        images = [
            (self.gray, f"{self.image_name}Original Image", {}),
            (
                harris_1,
                f"{self.image_name}Harris (Block=2, k=0.04)\nFeatures: {harris_stats_1['num_features']}",
                harris_stats_1,
            ),
            (
                harris_2,
                f"{self.image_name}Harris (Block=3, k=0.04)\nFeatures: {harris_stats_2['num_features']}",
                harris_stats_2,
            ),
            (
                harris_3,
                f"{self.image_name}Harris (Block=5, k=0.04)\nFeatures: {harris_stats_3['num_features']}",
                harris_stats_3,
            ),
            (
                sift_1,
                f"{self.image_name}SIFT (N=100)\nFeatures: {sift_stats_1['num_features']}",
                sift_stats_1,
            ),
            (
                sift_2,
                f"{self.image_name}SIFT (N=200)\nFeatures: {sift_stats_2['num_features']}",
                sift_stats_2,
            ),
            (
                sift_3,
                f"{self.image_name}SIFT (All)\nFeatures: {sift_stats_3['num_features']}",
                sift_stats_3,
            ),
            (
                fast_1,
                f"{self.image_name}FAST (Threshold=10)\nFeatures: {fast_stats_1['num_features']}",
                fast_stats_1,
            ),
            (
                fast_2,
                f"{self.image_name}FAST (Threshold=25)\nFeatures: {fast_stats_2['num_features']}",
                fast_stats_2,
            ),
        ]

        for ax, (img, title, stats) in zip(axes.flat, images):
            if len(img.shape) == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()

        if output_dir and save_path:
            plt.savefig(output_dir / save_path, dpi=300, bbox_inches="tight")

            # Save individual images
            for img, title, stats in images:
                filename = title.split("\n")[0].replace(" ", "_").lower() + ".png"
                cv2.imwrite(str(output_dir / filename), img)

            # Save statistics to JSON
            stats_file = (
                output_dir / f"{self.image_name.strip().lower()}_statistics.json"
            )
            with open(stats_file, "w") as f:
                json.dump(all_stats, f, indent=2)

            print(f"Visualization saved to {output_dir / save_path}")
            print(f"Statistics saved to {stats_file}")

        plt.close(fig)

        return all_stats


def main():
    """Main function to demonstrate feature detection"""
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Apply feature detection
    print("Applying feature detection methods...")
    camera_detector = FeatureDetector(camera, image_name="Camera ")
    coin_detector = FeatureDetector(coin, image_name="Coin ")
    checkerboard_detector = FeatureDetector(checkerboard, image_name="Checkerboard ")
    astronaut_detector = FeatureDetector(astronaut, image_name="Astronaut ")
    chelsea_detector = FeatureDetector(chelsea, image_name="Chelsea ")
    personal_image_detector = FeatureDetector(
        personal_image, image_name="Personal_Image "
    )

    # Visualize results and collect statistics
    all_statistics = {}

    camera_stats = camera_detector.visualize_all_features(
        output_dir=output_dir, save_path="camera_feature_detection_results.png"
    )
    all_statistics.update(camera_stats)

    coin_stats = coin_detector.visualize_all_features(
        output_dir=output_dir, save_path="coin_feature_detection_results.png"
    )
    all_statistics.update(coin_stats)

    checkerboard_stats = checkerboard_detector.visualize_all_features(
        output_dir=output_dir, save_path="checkerboard_feature_detection_results.png"
    )
    all_statistics.update(checkerboard_stats)

    astronaut_stats = astronaut_detector.visualize_all_features(
        output_dir=output_dir, save_path="astronaut_feature_detection_results.png"
    )
    all_statistics.update(astronaut_stats)

    chelsea_stats = chelsea_detector.visualize_all_features(
        output_dir=output_dir, save_path="chelsea_feature_detection_results.png"
    )
    all_statistics.update(chelsea_stats)

    personal_image_stats = personal_image_detector.visualize_all_features(
        output_dir=output_dir, save_path="personal_image_feature_detection_results.png"
    )
    all_statistics.update(personal_image_stats)

    # Save combined statistics
    combined_stats_file = output_dir / "all_statistics.json"
    with open(combined_stats_file, "w") as f:
        json.dump(all_statistics, f, indent=2)

    print(f"\nResults saved in: {output_dir}")
    print(f"Combined statistics saved to: {combined_stats_file}")


if __name__ == "__main__":
    main()

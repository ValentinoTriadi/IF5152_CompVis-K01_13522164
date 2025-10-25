# Nama: Valentino Chryslie Triadi
# NIM: 13522164
# Fitur unik: Geometric Transformations

# Import dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import data
import json
import csv

# Load image example from skimage
checkerboard = data.checkerboard()

# Load image personal
personal_image_path = Path(__file__).parent / "personal_image.png"
personal_image = cv2.imread(str(personal_image_path))
personal_image_rgb = cv2.cvtColor(personal_image, cv2.COLOR_BGR2RGB)


class GeometricTransform:
    """Class for various geometric transformation operations"""

    def __init__(self, image, image_name="image"):
        """
        Initialize with an image

        Args:
            image: Input image (numpy array)
            image_name: Name prefix for saved files
        """
        self.original = image
        self.image_name = image_name
        if self.original is None:
            raise ValueError(f"Cannot load image {image}")

        # Handle grayscale images
        if len(self.original.shape) == 2:
            self.gray = self.original
            self.original_rgb = cv2.cvtColor(self.original, cv2.COLOR_GRAY2RGB)
        else:
            self.gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
            self.original_rgb = self.original

        self.height, self.width = self.gray.shape[:2]

        # Store transformation parameters
        self.transform_params = {}

    def translate(self, tx, ty):
        """
        Translate (shift) the image

        Args:
            tx: Translation in x direction (pixels)
            ty: Translation in y direction (pixels)

        Returns:
            Translated image and transformation matrix
        """
        # Translation matrix
        M = np.float32([[1, 0, tx], [0, 1, ty]])

        # Apply translation
        translated = cv2.warpAffine(self.original_rgb, M, (self.width, self.height))

        return translated, M

    def rotate(self, angle, center=None, scale=1.0):
        """
        Rotate the image

        Args:
            angle: Rotation angle in degrees (counter-clockwise)
            center: Center of rotation (default: image center)
            scale: Scaling factor

        Returns:
            Rotated image and transformation matrix
        """
        if center is None:
            center = (self.width // 2, self.height // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # Apply rotation
        rotated = cv2.warpAffine(self.original_rgb, M, (self.width, self.height))

        return rotated, M

    def affine_transform(self, src_points, dst_points):
        """
        Apply affine transformation using three point correspondences

        Args:
            src_points: Three source points
            dst_points: Three destination points

        Returns:
            Transformed image and transformation matrix
        """
        # Get affine transformation matrix
        M = cv2.getAffineTransform(src_points, dst_points)

        # Apply transformation
        transformed = cv2.warpAffine(self.original_rgb, M, (self.width, self.height))

        return transformed, M

    def perspective_transform(self, src_points, dst_points):
        """
        Apply perspective transformation using four point correspondences

        Args:
            src_points: Four source points
            dst_points: Four destination points

        Returns:
            Transformed image and transformation matrix
        """
        # Get perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply transformation
        transformed = cv2.warpPerspective(
            self.original_rgb, M, (self.width, self.height)
        )

        return transformed, M

    def calibrate_checkerboard(self, pattern_size=(7, 7), square_size=1.0):
        """
        Detect and visualize checkerboard pattern for camera calibration

        Args:
            pattern_size: Number of inner corners (width, height)
            square_size: Size of checkerboard square in world units

        Returns:
            Visualization image and calibration parameters
        """
        # Generate 3D points for the checkerboard
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(
            -1, 2
        )
        objp *= square_size

        # Try to find checkerboard corners
        ret, corners = cv2.findChessboardCorners(self.gray, pattern_size, None)

        calibration_result = {
            "pattern_found": ret,
            "pattern_size": list(pattern_size),
            "square_size": square_size,
            "num_corners": pattern_size[0] * pattern_size[1],
        }

        vis_img = self.original_rgb.copy()

        if ret:
            # Refine corner positions for sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(
                self.gray, corners, (11, 11), (-1, -1), criteria
            )

            # Draw corners on visualization
            vis_img = cv2.drawChessboardCorners(
                vis_img, pattern_size, corners_refined, ret
            )

            calibration_result["corners_found"] = len(corners_refined)
            calibration_result["first_corner"] = corners_refined[0].tolist()
            calibration_result["last_corner"] = corners_refined[-1].tolist()

            # Calculate average corner response (quality metric)
            corner_responses = []
            for corner in corners_refined:
                x, y = int(corner[0][0]), int(corner[0][1])
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Simple corner quality based on local variance
                    window = self.gray[
                        max(0, y - 5) : min(self.height, y + 5),
                        max(0, x - 5) : min(self.width, x + 5),
                    ]
                    if window.size > 0:
                        corner_responses.append(float(np.std(window)))

            if corner_responses:
                calibration_result["avg_corner_quality"] = float(
                    np.mean(corner_responses)
                )
                calibration_result["min_corner_quality"] = float(
                    np.min(corner_responses)
                )
                calibration_result["max_corner_quality"] = float(
                    np.max(corner_responses)
                )
        else:
            calibration_result["corners_found"] = 0
            calibration_result["reason"] = (
                "Pattern not detected - ensure checkerboard is visible and well-lit"
            )

        return vis_img, calibration_result

    def visualize_all_transforms(self, output_dir=None, save_path=None):
        """
        Visualize all geometric transformations (Translation, Rotation, Affine, Perspective)
        with checkerboard calibration applied to each transformed image

        Args:
            output_dir: Directory to save the individual images
            save_path: Path to save the combined visualization
        """
        h, w = self.height, self.width

        # Apply transformations and collect parameters
        # 1. Translation
        translated, trans_matrix = self.translate(50, 30)

        # 2. Rotations (3 different angles)
        rotated_15, rot_matrix_15 = self.rotate(15)
        rotated_45, rot_matrix_45 = self.rotate(45)
        rotated_90, rot_matrix_90 = self.rotate(90)

        # 3. Affine transformations (2 variations)
        src_pts1 = np.float32([[50, 50], [w - 50, 50], [50, h - 50]])
        dst_pts1 = np.float32([[30, 70], [w - 30, 30], [70, h - 70]])
        affine_1, affine_matrix_1 = self.affine_transform(src_pts1, dst_pts1)

        src_pts2 = np.float32([[0, 0], [w, 0], [0, h]])
        dst_pts2 = np.float32([[w * 0.1, 0], [w * 0.9, h * 0.1], [w * 0.1, h * 0.9]])
        affine_2, affine_matrix_2 = self.affine_transform(src_pts2, dst_pts2)

        # 4. Perspective transformations (2 variations)
        src_pts_persp1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts_persp1 = np.float32(
            [[w * 0.1, 0], [w * 0.9, h * 0.05], [w * 0.85, h * 0.95], [w * 0.05, h]]
        )
        persp_1, persp_matrix_1 = self.perspective_transform(
            src_pts_persp1, dst_pts_persp1
        )

        dst_pts_persp2 = np.float32(
            [
                [w * 0.2, h * 0.1],
                [w * 0.8, h * 0.15],
                [w * 0.9, h * 0.9],
                [w * 0.15, h * 0.85],
            ]
        )
        persp_2, persp_matrix_2 = self.perspective_transform(
            src_pts_persp1, dst_pts_persp2
        )

        # Apply checkerboard calibration to ORIGINAL image
        calib_vis_original, calib_params_original = self.calibrate_checkerboard()

        # Apply checkerboard calibration to ALL transformed images
        # Create transformer objects for each transformed image and apply calibration
        transformed_images_data = [
            (translated, "Translation", trans_matrix),
            (rotated_15, "Rotation_15", rot_matrix_15),
            (rotated_45, "Rotation_45", rot_matrix_45),
            (rotated_90, "Rotation_90", rot_matrix_90),
            (affine_1, "Affine_1", affine_matrix_1),
            (affine_2, "Affine_2", affine_matrix_2),
            (persp_1, "Perspective_1", persp_matrix_1),
            (persp_2, "Perspective_2", persp_matrix_2),
        ]

        calibration_results = {}

        # Apply calibration to each transformed image
        for (
            transformed_img,
            transform_name,
            transform_matrix,
        ) in transformed_images_data:
            # Create temporary transformer for this transformed image
            temp_transformer = GeometricTransform(
                transformed_img, f"{self.image_name}_{transform_name}"
            )

            # Apply checkerboard calibration
            calib_vis, calib_params = temp_transformer.calibrate_checkerboard()

            # Store calibration visualization and parameters
            calibration_results[transform_name] = {
                "calibrated_image": calib_vis,
                "calibration_params": calib_params,
            }

        # Store all transformation parameters with calibration data
        all_params = {
            f"{self.image_name}Translation": {
                "type": "Translation",
                "tx": 50,
                "ty": 30,
                "matrix": trans_matrix.tolist(),
                "calibration": calibration_results.get("Translation", {}).get(
                    "calibration_params", {}
                ),
            },
            f"{self.image_name}Rotation_15": {
                "type": "Rotation",
                "angle": 15,
                "center": [w // 2, h // 2],
                "matrix": rot_matrix_15.tolist(),
                "calibration": calibration_results.get("Rotation_15", {}).get(
                    "calibration_params", {}
                ),
            },
            f"{self.image_name}Rotation_45": {
                "type": "Rotation",
                "angle": 45,
                "center": [w // 2, h // 2],
                "matrix": rot_matrix_45.tolist(),
                "calibration": calibration_results.get("Rotation_45", {}).get(
                    "calibration_params", {}
                ),
            },
            f"{self.image_name}Rotation_90": {
                "type": "Rotation",
                "angle": 90,
                "center": [w // 2, h // 2],
                "matrix": rot_matrix_90.tolist(),
                "calibration": calibration_results.get("Rotation_90", {}).get(
                    "calibration_params", {}
                ),
            },
            f"{self.image_name}Affine_1": {
                "type": "Affine",
                "src_points": src_pts1.tolist(),
                "dst_points": dst_pts1.tolist(),
                "matrix": affine_matrix_1.tolist(),
                "calibration": calibration_results.get("Affine_1", {}).get(
                    "calibration_params", {}
                ),
            },
            f"{self.image_name}Affine_2": {
                "type": "Affine",
                "src_points": src_pts2.tolist(),
                "dst_points": dst_pts2.tolist(),
                "matrix": affine_matrix_2.tolist(),
                "calibration": calibration_results.get("Affine_2", {}).get(
                    "calibration_params", {}
                ),
            },
            f"{self.image_name}Perspective_1": {
                "type": "Perspective",
                "src_points": src_pts_persp1.tolist(),
                "dst_points": dst_pts_persp1.tolist(),
                "matrix": persp_matrix_1.tolist(),
                "calibration": calibration_results.get("Perspective_1", {}).get(
                    "calibration_params", {}
                ),
            },
            f"{self.image_name}Perspective_2": {
                "type": "Perspective",
                "src_points": src_pts_persp1.tolist(),
                "dst_points": dst_pts_persp2.tolist(),
                "matrix": persp_matrix_2.tolist(),
                "calibration": calibration_results.get("Perspective_2", {}).get(
                    "calibration_params", {}
                ),
            },
            f"{self.image_name}Calibration_Original": calib_params_original,
        }

        # Create visualization (3x3 grid) - use CALIBRATED images
        fig, axes = plt.subplots(3, 3, figsize=(25, 25))

        # Use calibrated images for transformed images
        images = [
            (calib_vis_original, f"{self.image_name}Original Image (Calibrated)"),
            (
                calibration_results.get("Translation", {}).get(
                    "calibrated_image", translated
                ),
                f"{self.image_name}Translation (Calibrated)",
            ),
            (
                calibration_results.get("Rotation_15", {}).get(
                    "calibrated_image", rotated_15
                ),
                f"{self.image_name}Rotation 15° (Calibrated)",
            ),
            (
                calibration_results.get("Rotation_45", {}).get(
                    "calibrated_image", rotated_45
                ),
                f"{self.image_name}Rotation 45° (Calibrated)",
            ),
            (
                calibration_results.get("Rotation_90", {}).get(
                    "calibrated_image", rotated_90
                ),
                f"{self.image_name}Rotation 90° (Calibrated)",
            ),
            (
                calibration_results.get("Affine_1", {}).get(
                    "calibrated_image", affine_1
                ),
                f"{self.image_name}Affine 1 (Calibrated)",
            ),
            (
                calibration_results.get("Affine_2", {}).get(
                    "calibrated_image", affine_2
                ),
                f"{self.image_name}Affine 2 (Calibrated)",
            ),
            (
                calibration_results.get("Perspective_1", {}).get(
                    "calibrated_image", persp_1
                ),
                f"{self.image_name}Perspective 1 (Calibrated)",
            ),
            (
                calibration_results.get("Perspective_2", {}).get(
                    "calibrated_image", persp_2
                ),
                f"{self.image_name}Perspective 2 (Calibrated)",
            ),
        ]

        for ax, (img, title) in zip(axes.flat, images):
            ax.imshow(img)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()

        if output_dir and save_path:
            plt.savefig(output_dir / save_path, dpi=300, bbox_inches="tight")

            # Save individual images
            for img, title in images:
                filename = title.replace(" ", "_").lower() + ".png"
                # Convert RGB to BGR for cv2.imwrite
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_dir / filename), img_bgr)

            # Save calibration visualization
            calib_filename = f"{self.image_name.replace(' ', '_').lower()}calibration_checkerboard.png"
            calib_bgr = cv2.cvtColor(calib_vis_original, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / calib_filename), calib_bgr)

            # Save transformation parameters to JSON
            params_file = (
                output_dir / f"{self.image_name.strip().lower()}_transform_params.json"
            )
            with open(params_file, "w") as f:
                json.dump(all_params, f, indent=2)

            # Save transformation parameters to CSV
            csv_file = (
                output_dir / f"{self.image_name.strip().lower()}_transform_params.csv"
            )
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Transform_Name",
                        "Type",
                        "Parameters",
                        "Matrix",
                        "Pattern_Detected",
                        "Corners_Found",
                    ]
                )

                for name, params in all_params.items():
                    trans_type = params.get("type", "Calibration")
                    calib = params.get("calibration", {})
                    pattern_detected = calib.get("pattern_found", "N/A")
                    corners_found = calib.get("corners_found", "N/A")

                    if trans_type == "Translation":
                        param_str = f"tx={params['tx']}, ty={params['ty']}"
                        matrix_str = str(params["matrix"])
                    elif trans_type == "Rotation":
                        param_str = (
                            f"angle={params['angle']}°, center={params['center']}"
                        )
                        matrix_str = str(params["matrix"])
                    elif trans_type in ["Affine", "Perspective"]:
                        param_str = f"src_pts={len(params['src_points'])} pts, dst_pts={len(params['dst_points'])} pts"
                        matrix_str = str(params["matrix"])
                    else:  # Calibration
                        param_str = f"pattern_size={params.get('pattern_size')}, square_size={params.get('square_size')}"
                        matrix_str = f"pattern_found={params.get('pattern_found')}"
                        pattern_detected = params.get("pattern_found", "N/A")
                        corners_found = params.get("corners_found", "N/A")

                    writer.writerow(
                        [
                            name,
                            trans_type,
                            param_str,
                            matrix_str,
                            pattern_detected,
                            corners_found,
                        ]
                    )

            print(f"Visualization saved to {output_dir / save_path}")
            print(f"Parameters saved to {params_file} and {csv_file}")

        plt.close(fig)

        return all_params


def main():
    """Main function to demonstrate geometric transformations"""
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Apply geometric transformations
    print("Applying geometric transformations...")
    checkerboard_transformer = GeometricTransform(
        checkerboard, image_name="Checkerboard "
    )

    # Collect all transformation parameters
    all_transform_params = {}

    # Visualize results and collect parameters
    checkerboard_params = checkerboard_transformer.visualize_all_transforms(
        output_dir=output_dir, save_path="checkerboard_geometric_transforms.png"
    )
    all_transform_params.update(checkerboard_params)

    # Process personal image if available
    if personal_image is not None:
        personal_transformer = GeometricTransform(
            personal_image_rgb, image_name="Personal_Image "
        )
        personal_params = personal_transformer.visualize_all_transforms(
            output_dir=output_dir, save_path="personal_image_geometric_transforms.png"
        )
        all_transform_params.update(personal_params)

    # Save combined transformation parameters
    combined_params_file = output_dir / "all_transform_params.json"
    with open(combined_params_file, "w") as f:
        json.dump(all_transform_params, f, indent=2)

    # Save combined CSV
    combined_csv_file = output_dir / "all_transform_params.csv"
    with open(combined_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Transform_Name", "Image", "Type", "Parameters"])

        for name, params in all_transform_params.items():
            trans_type = params.get("type", "Calibration")

            # Extract image name from transform name
            img_name = name.split(trans_type)[0].strip()

            if trans_type == "Translation":
                param_str = f"tx={params['tx']}, ty={params['ty']}"
            elif trans_type == "Rotation":
                param_str = f"angle={params['angle']}°"
            elif trans_type in ["Affine", "Perspective"]:
                param_str = f"Points: {len(params['src_points'])} correspondences"
            else:  # Calibration
                param_str = f"Pattern: {params.get('pattern_size')}, Found: {params.get('pattern_found')}"

            writer.writerow([name, img_name, trans_type, param_str])

    print(f"\nResults saved in: {output_dir}")
    print(
        f"Combined parameters saved to: {combined_params_file} and {combined_csv_file}"
    )


if __name__ == "__main__":
    main()

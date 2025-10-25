# Computer Vision Individual Assignment

## IF5152 - Tugas Individu CV

**Name:** Valentino Chryslie Triadi  
**NIM:** 13522164

---

## Project Structure

```
Valentino Chryslie Triadi_13522164_IF5152_TugasIndividuCV/
├── 01_filtering/
│   ├── filtering.py
│   ├── personal_image.png
│   ├── Result.md
│   └── output/
├── 02_edge/
│   ├── edge_detection.py
│   ├── personal_image.png
│   ├── Result.md
│   └── output/
├── 03_featurepoints/
│   ├── feature_detection.py
│   ├── personal_image.png
│   ├── Result.md
│   └── output/
├── 04_geometry/
│   ├── geometric_transforms.py
│   ├── personal_image.png
│   ├── Result.md
│   └── output/
└── README.md
```

---

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---


## Usage

### Run All Modules

```bash
python main.py
```

> Note: You can run each module independently by navigating to its directory and executing the corresponding script.

### 1. Image Filtering (01_filtering)

Implements various image filtering techniques:

- Gaussian Blur
  - Kernel Size 5x5, 9x9, 25x25
  - Sigma 2.0, 10.0
- Median Filter
  - Kernel Size 5x5, 9x9, 25x25

**Run:**

```bash
cd 01_filtering
python filtering.py
```

**Features:**

- Multiple filtering methods with different parameters
- Comparative visualization
- Results saved in `output/` directory

---

### 2. Edge Detection (02_edge)

Implements various edge detection algorithms:

- Sobel Edge Detection
  - Kernel Size 3x3, 5x5, 7x7, 9x9, 15x15, 25x25
- Canny Edge Detection
  - Thresholds: 30-100, 50-150, 100-200, 150-250

**Run:**

```bash
cd 02_edge
python edge_detection.py
```

**Features:**

- Multiple edge detection methods with different parameters
- Comparative visualization
- Results saved in `output/` directory

---

### 3. Feature Point Detection (03_featurepoints)

Implements various feature detection methods:

- Harris Corner Detection
  - Block 2, k 0.04
  - Block 3, k 0.04
  - Block 5, k 0.06
- SIFT (Scale-Invariant Feature Transform)
  - N 100, 200, All
- FAST (Features from Accelerated Segment Test)
  - Thresholds 10, 25

**Run:**

```bash
cd 03_featurepoints
python feature_detection.py
```

**Features:**

- Multiple feature detection methods with different parameters
- Comparative visualization
- Results saved in `output/` directory

---

### 4. Geometric Transformations (04_geometry)

Implements various geometric transformations:

- Translation
  - tx 50, ty 30
- Rotation
  - 15°, 45°, 90°
- Scaling
- Shearing
- Flipping (Horizontal/Vertical)
- Affine Transformation
- Perspective Transformation

**Run:**

```bash
cd 04_geometry
python geometric_transforms.py
```

**Features:**

- Basic transformations
- Advanced transformations (Affine, Perspective)
- Comprehensive visualization
- Results saved in `output/` directory

---

## Output

Each module generates:

1. **Processed images** - Results of applying various algorithms
2. **Visualizations** - Comparative plots showing all methods

All outputs are saved in respective `output/` directories.

---

## Notes

- Each script can be run independently
- You can replace sample images with your own images
- Results are saved as high-resolution PNG files (300 DPI)

---

## Customization

To use your own images, change `personal_image.png` in each module's directory or modify the image path in the scripts:

```python
# Load image personal
personal_image_path = Path(__file__).parent / "personal_image.png"
personal_image = cv2.imread(str(personal_image_path))
```

---

## Author

**Valentino Chryslie Triadi**  
NIM: 13522164

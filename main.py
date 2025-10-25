import importlib

filtering = importlib.import_module('01_filtering.filtering')
edge_detection = importlib.import_module('02_edge.edge_detection')
feature_point_detection = importlib.import_module('03_featurepoints.feature_detection')
geometric_transforms = importlib.import_module('04_geometry.geometric_transforms')

def main():
    """Main function to run all modules"""

    print("Running all modules...\n")

    print("\n1. Image Filtering Module")
    filtering.main()

    print("\n2. Edge Detection Module")
    edge_detection.main()

    print("\n3. Feature Point Detection Module")
    feature_point_detection.main()

    print("\n4. Geometric Transforms Module")
    geometric_transforms.main()


if __name__ == "__main__":
    main()

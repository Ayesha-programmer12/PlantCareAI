
import os
import random
import numpy as np
from PIL import Image, ImageFilter

DATASET_DIR = r"D:\PlantCareAI\dataset\prepared"
MODEL_DIR = r"D:\PlantCareAI\model"

IMAGE_SIZE = (32, 32)
random.seed(42)
np.random.seed(42)


def extract_features(image_path):
    """Extract lightweight color and texture features."""

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    pixels = np.asarray(image, dtype=np.uint8)

    features = []

    # RGB color histograms
    for channel in range(3):
        histogram, _ = np.histogram(
            pixels[:, :, channel],
            bins=16,
            range=(0, 256)
        )
        features.extend(histogram / histogram.sum())

    # Grayscale histogram
    gray = image.convert("L")
    gray_pixels = np.asarray(gray)

    histogram, _ = np.histogram(
        gray_pixels,
        bins=16,
        range=(0, 256)
    )
    features.extend(histogram / histogram.sum())

    # Texture information using edge detection
    edges = image.filter(ImageFilter.FIND_EDGES)
    edge_pixels = np.asarray(edges.convert("L"))

    histogram, _ = np.histogram(
        edge_pixels,
        bins=8,
        range=(0, 256)
    )
    features.extend(histogram / histogram.sum())

    return np.array(features, dtype=np.float32)


def main():

    classes = sorted([
        folder for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    ])

    if not classes:
        print("No dataset categories found!")
        return

    features = []
    labels = []

    print("Extracting image features...\n")

    for class_index, class_name in enumerate(classes):

        folder = os.path.join(DATASET_DIR, class_name)

        images = [
            file for file in os.listdir(folder)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)

        for image_name in images:

            image_path = os.path.join(folder, image_name)

            try:
                feature = extract_features(image_path)
                features.append(feature)
                labels.append(class_index)

            except Exception as error:
                print("Skipped:", image_name, error)

        print(f"{class_name}: processed")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    if len(X) == 0:
        print("No images could be processed!")
        return

    # Split each category into training and testing data
    train_indices = []
    test_indices = []

    for class_index in range(len(classes)):

        indices = np.where(y == class_index)[0]
        np.random.shuffle(indices)

        split = int(len(indices) * 0.80)

        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    # Normalize features using training data only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print("\nTraining lightweight classifier...")

    # Three nearest neighbors
    correct = 0

    for i in range(len(X_test)):

        distances = np.sum(
            (X_train - X_test[i]) ** 2,
            axis=1
        )

        nearest = np.argpartition(distances, 3)[:3]
        predictions = y_train[nearest]

        predicted_class = np.bincount(
            predictions,
            minlength=len(classes)
        ).argmax()

        if predicted_class == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test) * 100

    # Save the trained model
    os.makedirs(MODEL_DIR, exist_ok=True)

    np.savez_compressed(
        os.path.join(MODEL_DIR, "plant_model.npz"),
        X_train=X_train,
        y_train=y_train,
        classes=np.array(classes),
        mean=mean,
        std=std
    )

    print("\nTraining completed!")
    print("Training images:", len(X_train))
    print("Testing images:", len(X_test))
    print(f"Test accuracy: {accuracy:.2f}%")
    print("Model saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()

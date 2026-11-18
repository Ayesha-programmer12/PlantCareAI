
from flask import Flask, render_template, request
from PIL import Image, ImageFilter
from pathlib import Path
from uuid import uuid4
import numpy as np

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "model" / "plant_model.npz"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Load the trained model
MODEL = None

if MODEL_PATH.exists():
    with np.load(MODEL_PATH, allow_pickle=False) as data:
        MODEL = {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "classes": data["classes"],
            "mean": data["mean"],
            "std": data["std"]
        }


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((32, 32))

    pixels = np.asarray(image, dtype=np.uint8)
    features = []

    for channel in range(3):
        histogram, _ = np.histogram(
            pixels[:, :, channel],
            bins=16,
            range=(0, 256)
        )
        features.extend(histogram / histogram.sum())

    gray_pixels = np.asarray(image.convert("L"))

    histogram, _ = np.histogram(
        gray_pixels,
        bins=16,
        range=(0, 256)
    )
    features.extend(histogram / histogram.sum())

    edges = image.filter(ImageFilter.FIND_EDGES)
    edge_pixels = np.asarray(edges.convert("L"))

    histogram, _ = np.histogram(
        edge_pixels,
        bins=8,
        range=(0, 256)
    )
    features.extend(histogram / histogram.sum())

    return np.array(features, dtype=np.float32)


def predict_disease(image_path):

    if MODEL is None:
        raise RuntimeError("Trained model not found.")

    features = extract_features(image_path)

    features = (
        features - MODEL["mean"]
    ) / MODEL["std"]

    distances = np.sum(
        (MODEL["X_train"] - features) ** 2,
        axis=1
    )

    nearest = np.argpartition(distances, 3)[:3]
    predictions = MODEL["y_train"][nearest]

    votes = np.bincount(
        predictions,
        minlength=len(MODEL["classes"])
    )

    predicted_index = int(np.argmax(votes))
    predicted_class = str(MODEL["classes"][predicted_index])

    agreement = int(votes[predicted_index] / 3 * 100)

    return {
        "name": " ".join(predicted_class.replace("_", " ").split()),
        "plant": predicted_class.split("_")[0],
        "healthy": "healthy" in predicted_class.lower(),
        "agreement": agreement
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():

    if request.method == "GET":
        return render_template("detect.html")

    file = request.files.get("plant_image")

    if not file or file.filename == "":
        return render_template(
            "detect.html",
            error="Please select a leaf image."
        )

    if not allowed_file(file.filename):
        return render_template(
            "detect.html",
            error="Please upload a JPG, JPEG, or PNG image."
        )

    try:
        image = Image.open(file.stream)
        image.verify()

        file.stream.seek(0)
        image = Image.open(file.stream).convert("RGB")
        image.thumbnail((1200, 1200))

        filename = f"{uuid4().hex}.jpg"
        image_path = UPLOAD_FOLDER / filename
        image.save(image_path, "JPEG", quality=90)

    except Exception:
        return render_template(
            "detect.html",
            error="The file is not a valid image. Please try another."
        )

    try:
        prediction = predict_disease(image_path)

        return render_template(
            "result.html",
            image_url=f"/static/uploads/{filename}",
            prediction=prediction
        )

    except Exception as error:
        print("Prediction error:", error)

        return render_template(
            "detect.html",
            error="Prediction failed. Please check the trained model."
        )


if __name__ == "__main__":
    app.run(debug=True)

import sys
from pathlib import Path

from flask import Flask, render_template, request, abort
from werkzeug.utils import secure_filename

# Add project root to path so `src.` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.image_classifier.predictor import predict_image
from src.nutrition_engine.nutrition_lookup import NutritionLookup

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB max upload

UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

lookup = NutritionLookup()


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files.get("image")

    if image is None or image.filename == "":
        abort(400, "No image uploaded.")

    if not allowed_file(image.filename):
        abort(400, "Unsupported file type.")

    filename = secure_filename(image.filename)
    image_path = UPLOAD_FOLDER / filename
    image.save(image_path)

    food, confidence, is_food, top5 = predict_image(image_path)

    if not is_food:
        return render_template(
            "index.html",
            not_food=True,
            confidence=confidence,
        )

    nutrition = lookup.get_nutrition(food)

    return render_template(
        "index.html",
        food=food,
        confidence=confidence,
        nutrition=nutrition,
    )


if __name__ == "__main__":
    # debug=False for safety. Set to True only while developing locally.
    app.run(debug=False)
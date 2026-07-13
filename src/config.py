from pathlib import Path

# Project root (…/nut)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths
MODEL_PATH = PROJECT_ROOT / "food_classifier.pth"
NUTRITION_CSV = PROJECT_ROOT / "data" / "nutrition" / "nutrition_data.csv"
IMAGES_FOLDER = PROJECT_ROOT / "data" / "raw" / "food101" / "food-101" / "images"

# The 20 classes — order MUST stay fixed (it defines the model's label indices)
SELECTED_CLASSES = [
    "apple_pie",
    "hamburger",
    "caesar_salad",
    "cheesecake",
    "chicken_curry",
    "donuts",
    "french_fries",
    "fried_rice",
    "grilled_salmon",
    "hot_dog",
    "ice_cream",
    "omelette",
    "pancakes",
    "pizza",
    "ramen",
    "steak",
    "sushi",
    "tacos",
    "waffles",
    "chocolate_cake",
]
import torch
from PIL import Image
from torchvision import models, transforms

from src.config import SELECTED_CLASSES, MODEL_PATH
from src.image_classifier.model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(len(SELECTED_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# A second, un-fine-tuned general model used purely as a "does this look
# like food at all" sanity check. Our 40-class model has no way to express
# "this is a person" or "this is a screenshot" — it must always output one
# of its 40 foods. The general ImageNet model's 1000 classes cover ordinary
# objects/scenes too, so it gives a much better signal for rejecting clearly
# non-food images before the specialized model is trusted.
_general_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
_general_model.to(device)
_general_model.eval()

_IMAGENET_CATEGORIES = models.ResNet18_Weights.DEFAULT.meta["categories"]

# ImageNet classes that are genuine food/dishes/ingredients. Several of our
# 40 foods (samosa, bibimbap, ...) have no exact ImageNet match, so instead
# of requiring the top-1 guess to be one of these, we sum the probability
# mass across all of them — related dishes still land some weight nearby,
# while non-food images land almost none.
_FOOD_IMAGENET_NAMES = {
    "guacamole", "consomme", "hot pot", "trifle", "ice cream", "ice lolly",
    "French loaf", "bagel", "pretzel", "cheeseburger", "hotdog",
    "mashed potato", "head cabbage", "broccoli", "cauliflower", "zucchini",
    "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith",
    "strawberry", "orange", "lemon", "fig", "pineapple", "banana",
    "jackfruit", "custard apple", "pomegranate", "carbonara",
    "chocolate sauce", "dough", "meat loaf", "pizza", "potpie", "burrito",
    "red wine", "espresso", "eggnog",
}
_FOOD_INDICES = torch.tensor([
    i for i, name in enumerate(_IMAGENET_CATEGORIES)
    if name in _FOOD_IMAGENET_NAMES
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Tune these on your own images.
CONFIDENCE_THRESHOLD = 52.0     # top-1 must be at least this % ...
MARGIN_THRESHOLD = 15.0         # ... AND beat top-2 by at least this many %
FOOD_LIKELIHOOD_THRESHOLD = 0.05  # ... AND look at least this food-like overall


def _food_likelihood(image_tensor):
    """Combined probability the general ImageNet model puts on food classes."""
    with torch.no_grad():
        outputs = _general_model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return probs[_FOOD_INDICES].sum().item()


def predict_image(image_path):
    """
    Returns (food, confidence, is_food, top5).
    is_food is False when the image doesn't confidently match any class,
    or when it doesn't look food-like at all by a broader general check.
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idx = torch.topk(probs, 5)

    top_probs = [p.item() * 100 for p in top_probs[0]]
    top_names = [SELECTED_CLASSES[i.item()] for i in top_idx[0]]

    food = top_names[0]
    confidence = top_probs[0]
    margin = top_probs[0] - top_probs[1]

    is_food = (
        confidence >= CONFIDENCE_THRESHOLD
        and margin >= MARGIN_THRESHOLD
        and _food_likelihood(image) >= FOOD_LIKELIHOOD_THRESHOLD
    )

    top5 = list(zip(top_names, top_probs))
    return food, confidence, is_food, top5
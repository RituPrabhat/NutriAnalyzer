from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.image_classifier.model import get_model

# -----------------------------
# Classes
# -----------------------------
selected_classes = [
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
    "chocolate_cake"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
model = get_model(len(selected_classes))

model.load_state_dict(
    torch.load(
        "food_classifier.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    return (
        selected_classes[prediction.item()],
        confidence.item() * 100
    )
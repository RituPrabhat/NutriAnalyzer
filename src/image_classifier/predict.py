import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.nutrition_engine.nutrition_lookup import NutritionLookup
from model import get_model

# -----------------------------
# Classes (same order as training)
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

model.load_state_dict(torch.load("food_classifier.pth", map_location=device))

model.to(device)

model.eval()

print("Model Loaded!")
lookup = NutritionLookup()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# Image Path
# -----------------------------
image_path = Path("test.jpg")      # Change later

image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probabilities, 5)

predicted_index = top_indices[0][0].item()
predicted_food = selected_classes[predicted_index]
confidence = top_probs[0][0].item() * 100

print("\n==============================")
print("Prediction Result")
print("==============================")
print(f"Food       : {predicted_food}")
print(f"Confidence : {confidence:.2f}%")

nutrition = lookup.get_nutrition(predicted_food)

if nutrition:
    print("\nNutrition Facts")
    print("------------------------------")
    print(f"Calories : {nutrition['Calories']} kcal")
    print(f"Protein  : {nutrition['Protein']} g")
    print(f"Carbs    : {nutrition['Carbs']} g")
    print(f"Fat      : {nutrition['Fat']} g")

print("\n==============================")
print("Top 5 Predictions")
print("==============================")

for prob, idx in zip(top_probs[0], top_indices[0]):
    print(f"{selected_classes[idx]} : {prob.item()*100:.2f}%")
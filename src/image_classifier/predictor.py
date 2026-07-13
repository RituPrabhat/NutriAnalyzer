import torch
from PIL import Image
from torchvision import transforms

from src.config import SELECTED_CLASSES, MODEL_PATH
from src.image_classifier.model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(len(SELECTED_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Tune these on your own images.
CONFIDENCE_THRESHOLD = 40.0   # top-1 must be at least this % ...
MARGIN_THRESHOLD = 15.0       # ... AND beat top-2 by at least this many %


def predict_image(image_path):
    """
    Returns (food, confidence, is_food, top5).
    is_food is False when the image doesn't confidently match any class.
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

    is_food = confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD

    top5 = list(zip(top_names, top_probs))
    return food, confidence, is_food, top5
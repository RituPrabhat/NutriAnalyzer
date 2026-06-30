from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import Food101Dataset
from model import get_model

# -----------------------------
# Dataset Path
# -----------------------------
images_folder = Path("data/raw/food101/food-101/images")

print("Current Working Directory:", Path.cwd())
print("Images Folder:", images_folder)
print("Images Folder Exists:", images_folder.exists())

# -----------------------------
# Selected Classes
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

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Dataset
# -----------------------------
dataset = Food101Dataset(
    images_folder=images_folder,
    selected_classes=selected_classes,
    transform=transform
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Total Images:", len(dataset))
print("Total Batches:", len(train_loader))

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# Model
# -----------------------------
model = get_model(len(selected_classes))
model = model.to(device)

print("Model Loaded Successfully!")

# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# -----------------------------
# Training
# -----------------------------
print("\nTraining Started...\n")

for epoch in range(EPOCHS):

    print("=" * 50)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 50)

    model.train()

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 5 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(train_loader)

    print(f"\nEpoch {epoch + 1} Average Loss: {epoch_loss:.4f}\n")

# -----------------------------
# Save Model
# -----------------------------
print("=" * 50)
print("Training Complete!")
print("=" * 50)

torch.save(model.state_dict(), "food_classifier.pth")

print("\nModel Saved Successfully!")
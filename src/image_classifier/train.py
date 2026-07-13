import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.config import SELECTED_CLASSES, IMAGES_FOLDER, MODEL_PATH
from src.image_classifier.dataset import Food101Dataset
from src.image_classifier.model import get_model

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
VAL_SPLIT = 0.2

print("Images Folder:", IMAGES_FOLDER)
print("Images Folder Exists:", IMAGES_FOLDER.exists())

# -----------------------------
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -----------------------------
# Dataset + train/val split
# -----------------------------
full_dataset = Food101Dataset(
    images_folder=IMAGES_FOLDER,
    selected_classes=SELECTED_CLASSES,
    transform=train_transform,
)

if len(full_dataset) == 0:
    raise RuntimeError(
        f"No images found in {IMAGES_FOLDER}. "
        "Download the Food-101 dataset first."
    )

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(full_dataset, [train_size, val_size], generator)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total: {len(full_dataset)} | Train: {train_size} | Val: {val_size}")

# -----------------------------
# Device + Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = get_model(len(SELECTED_CLASSES)).to(device)

criterion = nn.CrossEntropyLoss()
trainable_params = list(model.layer4.parameters()) + list(model.fc.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)


def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(len(loader), 1), 100 * correct / max(total, 1)


# -----------------------------
# Training loop
# -----------------------------
print("\nTraining Started...\n")

for epoch in range(EPOCHS):
    print("=" * 50)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 50)

    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 5 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)} Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    print(
        f"\nEpoch {epoch + 1} | Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n"
    )

# -----------------------------
# Save
# -----------------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
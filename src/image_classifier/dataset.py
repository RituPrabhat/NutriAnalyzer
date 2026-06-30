from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class Food101Dataset(Dataset):

    def __init__(self, images_folder, selected_classes, transform=None):

        self.transform = transform
        self.images = []
        self.labels = []

        # Create label mapping
        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(selected_classes)
        }

        # Read images from selected classes
        for class_name in selected_classes:

            class_folder = Path(images_folder) / class_name

            if not class_folder.exists():
                continue

            for image_path in class_folder.glob("*.jpg"):

                self.images.append(image_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
import torch.nn as nn
from torchvision import models


def get_model(num_classes, unfreeze_layer4=True):

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # With more classes, the frozen backbone alone underfits — unfreeze
    # the last residual block so it can adapt its features too.
    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace the final layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model
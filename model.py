# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class AntiSpoofNet(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True, freeze_backbone=False):
        super(AntiSpoofNet, self).__init__()

        # Select backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
        elif backbone_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Backbone '{backbone_name}' not supported. Choose resnet18, resnet34, or resnet50.")

        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head for binary classification (live/spoof)
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Output single logit for BCEWithLogitsLoss
        )

        # Always train classifier layers
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

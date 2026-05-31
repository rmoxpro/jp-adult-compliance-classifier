from __future__ import annotations

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

CLASSES = ("mosaic", "other", "uncensored")
CROP_INPUT_SIZE = 300
CTX_INPUT_SIZE = 224


class TwoStreamClassifier(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES), pretrained: bool = True):
        super().__init__()
        b3_weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        b0_weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.crop_backbone = models.efficientnet_b3(weights=b3_weights)
        self.crop_feat_dim = self.crop_backbone.classifier[1].in_features
        self.crop_backbone.classifier = nn.Identity()
        self.ctx_backbone = models.efficientnet_b0(weights=b0_weights)
        self.ctx_feat_dim = self.ctx_backbone.classifier[1].in_features
        self.ctx_backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.crop_feat_dim + self.ctx_feat_dim, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, crop: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        crop_features = self.crop_backbone(crop)
        context_features = self.ctx_backbone(context)
        return self.classifier(torch.cat([crop_features, context_features], dim=1))


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_transforms():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crop_train = transforms.Compose([
        transforms.Resize((CROP_INPUT_SIZE, CROP_INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        norm,
    ])
    crop_val = transforms.Compose([
        transforms.Resize((CROP_INPUT_SIZE, CROP_INPUT_SIZE)),
        transforms.ToTensor(),
        norm,
    ])
    ctx_train = transforms.Compose([
        transforms.Resize((CTX_INPUT_SIZE, CTX_INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        norm,
    ])
    ctx_val = transforms.Compose([
        transforms.Resize((CTX_INPUT_SIZE, CTX_INPUT_SIZE)),
        transforms.ToTensor(),
        norm,
    ])
    return crop_train, crop_val, ctx_train, ctx_val


class TwoStreamManifestDataset(Dataset):
    def __init__(self, entries, pairs_dir, crop_transform, context_transform, draw_bbox: bool = False):
        self.entries = entries
        self.pairs_dir = pairs_dir
        self.crop_transform = crop_transform
        self.context_transform = context_transform
        self.draw_bbox = draw_bbox
        self.label_to_idx = {name: i for i, name in enumerate(CLASSES)}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        item = self.entries[idx]
        crop_path = self.pairs_dir / item["crop_path"]
        frame_path = self.pairs_dir / item["frame_path"]
        crop = cv2.imread(str(crop_path))
        frame = cv2.imread(str(frame_path))
        if crop is None:
            raise FileNotFoundError(crop_path)
        if frame is None:
            raise FileNotFoundError(frame_path)
        if self.draw_bbox and item.get("bbox_norm"):
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = item["bbox_norm"]
            cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 0, 255), 4)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop_tensor = self.crop_transform(Image.fromarray(crop_rgb))
        context_tensor = self.context_transform(Image.fromarray(frame_rgb))
        label = self.label_to_idx[item["label"]]
        return crop_tensor, context_tensor, label

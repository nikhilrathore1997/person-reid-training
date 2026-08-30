# ? BEST PERSON RE-ID TRAINING SETUP
# Includes: ResNet-50 backbone, ArcFace head, TripletLoss + CrossEntropyLoss, RandomIdentitySampler
# Uses torchvision + vanilla PyTorch (no external frameworks)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data.sampler import Sampler
import random
import numpy as np

# ---------------- CONFIG ---------------- #
DATA_DIR = "D:\Nikhil\My_project\train_improved"  # folder structure: /class_name/image.jpg
BATCH_SIZE = 128#64
NUM_EPOCHS = 50
LR = 3.5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P = 32  # identities per batch
K = 4   # images per identity
FEATURE_DIM = 512
# FEATURE_DIM = 256
MARGIN = 0.3
# --------------------------------------- #

# ---------------- TRANSFORMS ---------------- #
# transform = transforms.Compose([
#     transforms.Resize((256, 128)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])


transform = transforms.Compose([
                transforms.Resize((288, 144)),
                transforms.RandomCrop((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
                # RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
            ])
# ---------------- RANDOM IDENTITY SAMPLER ---------------- #
class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = self._build_index()
        self.pids = list(self.index_dic.keys())

    def _build_index(self):
        index_dic = {}
        for index, (_, label) in enumerate(self.data_source.samples):
            index_dic.setdefault(label, []).append(index)
        return index_dic

    def __iter__(self):
        batch_idxs = []
        random.shuffle(self.pids)
        for pid in self.pids:
            idxs = self.index_dic[pid]
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            else:
                idxs = random.sample(idxs, self.num_instances)
            batch_idxs.extend(idxs)
        return iter(batch_idxs)

    def __len__(self):
        return len(self.pids) * self.num_instances

# ---------------- TRIPLET LOSS ---------------- #
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, embeddings, labels):
        def _pairwise_distances(emb):
            dot = torch.matmul(emb, emb.t())
            norm = torch.diag(dot)
            dist = norm.unsqueeze(1) - 2 * dot + norm.unsqueeze(0)
            return torch.clamp(dist, min=1e-12).sqrt()

        dist = _pairwise_distances(embeddings)
        N = labels.size(0)
        loss = 0
        for i in range(N):
            anchor = embeddings[i]
            anchor_label = labels[i]
            pos_mask = (labels == anchor_label)
            neg_mask = (labels != anchor_label)

            pos_dist = dist[i][pos_mask].max()
            neg_dist = dist[i][neg_mask].min()
            loss += torch.relu(pos_dist - neg_dist + self.margin)

        return loss / N

#----------------BATCHHard Triplet Loss----------#

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        N = labels.size(0)
        loss = []

        for i in range(N):
            anchor_label = labels[i]
            dist = dist_mat[i]

            pos_mask = labels == anchor_label
            pos_mask[i] = False
            if pos_mask.sum() == 0:
                continue
            hardest_pos = dist[pos_mask].max()

            neg_mask = labels != anchor_label
            if neg_mask.sum() == 0:
                continue
            hardest_neg = dist[neg_mask].min()

            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss.append(triplet_loss)

        if len(loss) == 0:
            return torch.tensor(0.0, requires_grad=True).to(embeddings.device)
        return torch.stack(loss).mean()


# ---------------- MODEL ---------------- #
class ReIDModel(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super().__init__()
        base = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.backbone(x)
        x = self.pool(x).view(x.size(0), -1)
        feat = self.embedding(x)
        if return_embedding:
            return F.normalize(feat, dim=1)
        else:
            logits = self.classifier(feat)
            return feat, logits

# ---------------- LOAD DATA ---------------- #
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        sampler=RandomIdentitySampler(dataset, num_instances=K),
                        num_workers=4, drop_last=True)
num_classes = len(dataset.classes)
print("num_classes",num_classes)
# ---------------- TRAINING ---------------- #
model = ReIDModel(num_classes=num_classes, feature_dim=FEATURE_DIM).to(DEVICE)
criterion_ce = nn.CrossEntropyLoss()
# criterion_tri = TripletLoss(margin=MARGIN)
criterion_tri = BatchHardTripletLoss(margin=MARGIN)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        features, logits = model(imgs)
        features_norm = F.normalize(features, p=2, dim=1)
        loss_ce = criterion_ce(logits, labels)
        # loss_tri = criterion_tri(features, labels)
        loss_tri = criterion_tri(features_norm, labels)
        loss = loss_ce + loss_tri

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    scheduler.step()
    acc = total_correct / total_samples
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss:.4f}, Acc: {acc:.4f}")

# ---------------- SAVE MODEL ---------------- #
torch.save(model.state_dict(), "reid_best_model_50_epochs_128_batch_512.pth")
print("? Training complete. Model saved.")

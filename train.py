import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.unet import UNet
from transforms import get_transforms
from utils import LaneDataset, dice_loss

# Hyperparams
EPOCHS = 30
BATCH_SIZE = 4
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Loader
train_ds = LaneDataset(
    image_dir="dataset/images",
    mask_dir="dataset/masks",
    transforms=get_transforms(),
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = UNet().to(device)
criterion_bce = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = 0.5 * criterion_bce(preds, masks) + 0.5 * dice_loss(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS} — Loss: {avg:.4f}")

# Save final model
torch.save(model.state_dict(), "lane_unet.pth")
print("[✅] Model saved to lane_unet.pth")

import os
import glob
import cv2
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------
#    Configurations
# -------------------
SEED = 42
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------
#     UNet Model
# -------------------
class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_block1 = UNetConvBlock(2, 32)
        self.enc_block2 = UNetConvBlock(32, 64)
        self.enc_block3 = UNetConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.mid_block = UNetConvBlock(128, 128)
        self.up_trans_3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec_block3 = UNetConvBlock(128 + 128, 64)
        self.up_trans_2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_block2 = UNetConvBlock(64 + 64, 32)
        self.up_trans_1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec_block1 = UNetConvBlock(32 + 32, 16)
        self.output_conv = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.enc_block1(x)
        x2 = self.pool(x1)
        x2 = self.enc_block2(x2)
        x3 = self.pool(x2)
        x3 = self.enc_block3(x3)
        x4 = self.pool(x3)
        x_mid = self.mid_block(x4)
        x = self.up_trans_3(x_mid)
        x = torch.cat([x, x3], dim=1)
        x = self.dec_block3(x)
        x = self.up_trans_2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec_block2(x)
        x = self.up_trans_1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec_block1(x)
        return self.output_conv(x)

# -------------------
#    Utility Functions
# -------------------
def rgb_to_hsv_torch(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = torch.max(rgb, dim=1).values
    minc = torch.min(rgb, dim=1).values
    v = maxc
    deltac = maxc - minc
    s = deltac / maxc.clamp(min=1e-5)
    h = torch.zeros_like(maxc)
    mask = deltac > 0
    r_mask = (maxc == r) & mask
    g_mask = (maxc == g) & mask
    b_mask = (maxc == b) & mask
    h[r_mask] = ((g - b) / deltac)[r_mask] % 6
    h[g_mask] = ((b - r) / deltac)[g_mask] + 2
    h[b_mask] = ((r - g) / deltac)[b_mask] + 4
    h = h / 6.0
    hsv = torch.stack([h, s, v], dim=1)
    return hsv

def hsv_to_rgb_torch(hsv):
    h, s, v = hsv[:, 0] * 6, hsv[:, 1], hsv[:, 2]
    c = s * v
    x = c * (1 - torch.abs((h % 2) - 1))
    m = v - c
    z = torch.zeros_like(h)
    cond = [(0 <= h) & (h < 1), (1 <= h) & (h < 2), (2 <= h) & (h < 3),
            (3 <= h) & (h < 4), (4 <= h) & (h < 5), (5 <= h) & (h < 6)]
    r, g, b = [torch.where(cond[i], c, x) for i in range(3)]
    return torch.stack([r + m, g + m, b + m], dim=1)

# -------------------
#      Main Logic
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--headless', action='store_true', help='Run without GUI previews')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_files = glob.glob('checkpoints/checkpoint_epoch_*.pth')
    latest_epoch = 0

    if checkpoint_files:
        try:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            latest_epoch = checkpoint['epoch']
            logger.info(f"Loaded checkpoint from epoch {latest_epoch}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    for epoch in range(latest_epoch + 1, latest_epoch + 1 + args.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            hsv_images = rgb_to_hsv_torch(images)
            noisy_hsv = hsv_images.clone()
            noisy_hsv[:, :2] += torch.randn_like(noisy_hsv[:, :2]) * 0.1
            noisy_hsv = torch.clamp(noisy_hsv, 0, 1)

            optimizer.zero_grad()
            predictions = model(noisy_hsv[:, :2])
            loss = criterion(predictions, hsv_images[:, :2])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    main()

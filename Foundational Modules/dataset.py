## This script will :
## have a custom PyTorch Dataset class specifically for the 2024 MSI dataset that will :
## load the single 3 channel RGB image
## load the 8 separate single channel MS images, stack them into a single [8, H, W] tensor
## returns the paired (RGB, MSI) tensors

from torch.utils.data import Dataset, Dataloader
import os

class MSIDataset(Dataset):
    def __init_(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [f for f in sorted(os.listdir(images_dir)) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG')]
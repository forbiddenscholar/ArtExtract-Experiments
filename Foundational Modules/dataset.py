## This script will :
## have a custom PyTorch Dataset class specifically for the 2024 MSI dataset that will :
## load the single 3 channel RGB image
## load the 8 separate single channel MS images, stack them into a single [8, H, W] tensor
## returns the paired (RGB, MSI) tensors

from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf

class MSIDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Get all RGB image files
        self.images = [f for f in sorted(os.listdir(images_dir)) 
                      if f.endswith(('.bmp', '.jpg', '.png', '.JPG'))]
        
        # Get the corresponding 8 MSI files for each RGB image
        self.msi_files = {}
        for img_name in self.images:
            base_name = img_name.split('_RGB')[0]
            msi_images = sorted([f for f in os.listdir(masks_dir) 
                                if f.startswith(base_name) and f != img_name])
            self.msi_files[img_name] = msi_images
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        
        # Load RGB image
        rgb_path = os.path.join(self.images_dir, img_name)
        rgb_image = Image.open(rgb_path).convert('RGB')
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        # Convert to tensor [3, H, W]
        # this is what 2024's project missed (it treated rgb as [1, H, W])
        rgb_tensor = tf.to_tensor(rgb_image)
        
        # Load 8 multispectral images
        msi_names = self.msi_files[img_name]
        msi_tensors = []
        
        for msi_name in msi_names:
            msi_path = os.path.join(self.masks_dir, msi_name)
            msi_image = Image.open(msi_path).convert('L')  # Grayscale (single channel)
            msi_tensor = tf.to_tensor(msi_image)  # Shape: [1, H, W]
            msi_tensors.append(msi_tensor)
        
        # Stack the 8 channels into [8, H, W]
        msi_tensor = torch.cat(msi_tensors, dim=0)
        
        ## add a check if returns 8 channels or not
        assert msi_tensor.shape[0] == 8, "MSI channels not equal to 8"
        
        return rgb_tensor, msi_tensor
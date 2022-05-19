import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms

# cd /kaggle/input/celeba-dataset/img_align_celeba/[000001-202599].jpg

class CustomImageDataset(Dataset):
    def __init__(self):
        self.transform = transforms.Normalize(0.5, 0.5)
        super().__init__()

    def __len__(self):
        return 202599

    def __getitem__(self, idx):
        img_path = f'/kaggle/input/celeba-dataset/img_align_celeba/{idx + 1:06d}.jpg'
        image = read_image(img_path)
        image = self.transform(image)
        return image

def get_train_dataloader(batch_size = 64, num_workers = 4):
    
    training_set = CustomImageDataset()

    train_dataloader = torch.utils.data.DataLoader(
        training_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    
    return train_dataloader


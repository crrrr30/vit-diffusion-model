import torch
from torch.utils.data import Dataset

from skimage import io
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        super().__init__()

    def __len__(self):
        return 30000

    def __getitem__(self, idx):
        img_path = f'/kaggle/input/celebahq-resized-256x256/celeba_hq_256/{idx:05d}.jpg'
        image = io.imread(img_path)
        image = self.transform(image)
        return image

def get_train_dataloader(batch_size, num_workers=2):
    
    training_set = CustomImageDataset()

    train_dataloader = torch.utils.data.DataLoader(
        training_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    
    return train_dataloader


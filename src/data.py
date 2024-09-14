import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb

SIZE = 256
# create another dataset
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC), 
                transforms.RandomHorizontalFlip(), # flip only the images that needs such colorization
                transforms.ToTensor() # convert to Tensors
                
            ])
        elif split == 'val':
            self.transforms = transforms.Compose([
              transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
              transforms.ToTensor() # convert to Tensors
            ])
        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img= self.transforms(img)
        
        #shape and flip
        img =img.permute(1,2,0).numpy()
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
      

        L = torch.tensor(img_lab[[0], ...] / 50. - 1.) # Between -1 and 1
        ab = img_lab([[1, 2], ...] / 110.) # Between -1 and 1

        return {'L': L.unsqueeze(0), 'ab': ab}

    def __len__(self):
        return len(self.paths)

def make_dataloaders(batch_size=16, n_workers=1, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

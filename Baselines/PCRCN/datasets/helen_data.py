from torch.utils.data import Dataset
from glob import glob
import numpy as np
import datasets.utility as util
import torchvision.transforms as transforms
from PIL import Image
class Helen(Dataset):
    def __init__(self) -> None:
        super(Helen,self).__init__()
        self.data_paths = glob('/home/zelin/csrnet/Dataset/helen/*')
        self.data_transforms = transforms.Compose([
        transforms.ToTensor()
    ]) 
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        hr = Image.open(self.data_paths[index])        
        lr_24 = hr.resize((24,24),Image.BILINEAR)
        
        return self.data_transforms(lr_24),self.data_transforms(hr)
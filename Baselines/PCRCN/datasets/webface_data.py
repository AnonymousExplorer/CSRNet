from torch.utils.data import Dataset
from glob import glob
import numpy as np
import datasets.utility as util
import torchvision.transforms as transforms
import cv2

class Webface_landmarks68(Dataset):
    def __init__(self) -> None:
        super(Webface_landmarks68,self).__init__()
        self.data_paths = glob('/home/zelin/csrnet/Dataset/webface_npy/*')
        self.data_transforms = transforms.Compose([
        transforms.ToTensor()
    ]) 
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        _,lr_24,lr_48,lr_96,hr = np.load(self.data_paths[index],allow_pickle=True)
        parsing = np.load(self.data_paths[index].replace('webface_npy','webface_parsing'))
        result = np.zeros((512,512,19))
        for w in range(512):
            for h in range(512):
                result[w,h,parsing[w,h]] = 1
        
        return self.data_transforms(lr_24),self.data_transforms(lr_48),self.data_transforms(lr_96),self.data_transforms(hr),np.transpose(result,[2,0,1])
    

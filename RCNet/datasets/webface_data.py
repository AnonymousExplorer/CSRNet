from torch.utils.data import Dataset
from glob import glob
import numpy as np
import datasets.utility as util
import torchvision.transforms as transforms
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
        lr_24,_,lr_48,lr_96,hr = np.load(self.data_paths[index],allow_pickle=True)
        lms = np.load(self.data_paths[index].replace('webface_npy','webface_landmarks'))
        tmp = [] 
        for lm in lms:
            # tmp.append(util._generate_one_heatmap((48,48),lm/4.0,1))
            tmp.append(util._generate_one_heatmap((192,192),lm,1))
        
        return self.data_transforms(lr_24),self.data_transforms(hr),np.stack(tmp,axis=0)
    

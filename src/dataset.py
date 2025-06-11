from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import PIL.ImageOps 
import torch


#自定义Dataset类，__getitem__(self,index)每次返回(img1, img2, 0/1)
class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None,should_invert=True, pos_rate=0.5):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        self.positive_rate = pos_rate
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs) #24类别中任选一个
        should_get_same_class = random.random() #保证同类样本约占一半
        if should_get_same_class < self.positive_rate:
            while True:
                #直到找到同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #直到找到非同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs*2)
    
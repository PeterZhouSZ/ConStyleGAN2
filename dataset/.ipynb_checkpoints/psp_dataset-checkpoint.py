import os 
import torch 
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np


"This is just to verify if my code is correct by reprodcing on celeba dataset"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_path, sem_path):
        img_files = os.listdir(img_path)
        img_files.sort()
        self.img_files = [os.path.join(img_path, item) for item in img_files]

        sem_files = os.listdir(sem_path)
        sem_files.sort()
        self.sem_files = [os.path.join(sem_path, item) for item in sem_files]

    def __getitem__(self,idx):
        img = Image.open(  self.img_files[idx] ).convert('RGB').resize((1024,1024))
        sem = Image.open(  self.sem_files[idx] ).resize((256,256), Image.NEAREST)

        img = ( (TF.to_tensor(img)-0.5)/0.5 )
        sem = torch.tensor( np.array(sem) ).unsqueeze(0).long()
        
        return img, sem


    def __len__(self):
        return len(self.img_files)
    
    
    
    
def get_dataloader(args, train):
    if train:
        img_path = '/home/code-base/scratch_space/Server/my_stylegan3/pixel2style2pixel/CelebAMask-HQ/train_img'
        sem_path = '/home/code-base/scratch_space/Server/my_stylegan3/pixel2style2pixel/CelebAMask-HQ/train_mask'
    else:
        img_path = '/home/code-base/scratch_space/Server/my_stylegan3/pixel2style2pixel/CelebAMask-HQ/test_img'
        sem_path = '/home/code-base/scratch_space/Server/my_stylegan3/pixel2style2pixel/CelebAMask-HQ/test_mask'
        
    dataset = Dataset(img_path, sem_path)
        
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0, drop_last=True, shuffle=True)
        

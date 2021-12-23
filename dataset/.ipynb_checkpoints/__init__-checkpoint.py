from .dataset import Dataset
from .memory_dataset import MemoryDataset
import os 
import time 
import torch 


def get_dataloader(args,train): 


    if train and os.path.exists('dataset/data'):
        print("preprocessed training data exists")
        print("MemoryDataset will be used, please make sure uncast and tweak code are implmeneted properly")
        time.sleep(1)
        dataset = MemoryDataset(args)
    else:
        dataset = Dataset(args,train)
    

    def collate_fn(data):
        target_img, target_seg, global_img, global_sem, global_pri, global_seg, crop_indicator, info = list(zip(*data))
        
        target_img = torch.stack(target_img)
        target_seg = torch.stack(target_seg)
        global_img = torch.stack(global_img)
        global_sem = torch.stack(global_sem)
        global_pri = torch.stack(global_pri)
        global_seg = torch.stack(global_seg)
        crop_indicator = torch.stack(crop_indicator)

        return  target_img, target_seg, global_img, global_sem, global_pri, global_seg, crop_indicator, info 


    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0, drop_last=True, shuffle=True)




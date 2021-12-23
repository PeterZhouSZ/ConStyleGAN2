import torch
import random
import pickle
import numpy as np
import torch.nn.functional as F
from .dataset import Dataset
from .dataset_helper import get_box, modify_x2y2 



   
    
class MemoryDataset(torch.utils.data.Dataset):
    """
    Dataset in dataset.py is actually final dataset, which can be used in training.
    But there is a lot of cpu process, so you can preprocess all data
    by running preprocess_dataset.py. This class will load all preprocessed data
    and make some small tweak to undo some extra processes implemented in preprocess_dataset.py  
    """
    def __init__(self,  args):

        self.args = args

                       
        with open('dataset/data', 'rb') as fp:
            training_data = pickle.load(fp)

        self.training_data = training_data
        
        self.safe_output = self[0]


    def uncast(self, datum):
        """
        Input datum is result of preprocess_dataset.py. Here we undo cast implemented in preprocessing:
        
        Note that: 

        1 cast function in preprocess_dataset.py and uncast function here should be exactly counteract 
        with each other. In other words, input to cast should be exactly same as output of uncast. 

        2 If you have enough memory you should avoid doing casting and uncasting to save cpu time 
            
        """
        target_img, target_seg, global_img, global_sem, global_pri, global_seg, crop_indicator, info = datum

        # first undo cast 
        target_img = ((target_img.float()/255) - 0.5 ) / 0.5 
        target_seg = target_seg.float()
        global_img = ((global_img.float()/255) - 0.5 ) / 0.5
        global_sem = global_sem.long()
        global_pri = ((global_pri.float()/255) - 0.5 ) / 0.5
        global_seg = global_seg.float()

        return target_img, target_seg, global_img, global_sem, global_pri, global_seg, crop_indicator, info


    def tweak(self, datum):
        """
        Here make some small changes, including:
            1,  randomly apply random crop on images
            2,  modify crop_indicator
            3,  adjuct new_box and new_size in info 

        
        Note that: 
        
        1 This tweak process should be very fast, otherwise there is no point to preprocess beforehand.
        2 The main reason why we need this function is we still want to apply random crop, and you can not 
          model and preprocess this 'randomness' in preprocess_dataset.py. Thus we crop and resize each data 
          slightly bigger and save them in preprocess_dataset.py, and apply random crop here. 
        3 If you do not want to apply random crop and want to save more cpu time, you should directly prepare 
          images to be 256 and 128 resolution in preprocess_dataset.py, and you can pass tweak code here.   

        """

        if random.random() > 0.5:
            return apply_crop(datum, self.args)
        else:
            return directly_resize(datum, self.args)

            
    def __getitem__(self, idx):

        # load preprocessed data (Check preprocess_dataset.py to see how these data being processed)
        datum = self.training_data[idx] 
        datum = self.uncast(datum)    
        try:
            return self.tweak(datum)
        except ValueError:
            return self.safe_output

        
    def __len__(self):
        return len(  self.training_data  )
            
 







def apply_crop(datum, args):

    target_img, target_seg, global_img, global_sem, global_pri, global_seg, crop_indicator, info = datum

    # apply random crop 
    y_start, x_start = random.randint(0,32), random.randint(0,32) # 32 = 292-256 (check preprocess_dataset.py) 
    target_img = target_img[ :, y_start:y_start+256, x_start:x_start+256 ]
    target_seg = target_seg[ :, y_start:y_start+256, x_start:x_start+256 ]

    y_start, x_start = int(y_start/2), int(x_start/2) # similar location in 128 and 256 res 
    global_img = global_img[ :, y_start:y_start+128, x_start:x_start+128 ]
    global_sem = global_sem[ :, y_start:y_start+128, x_start:x_start+128 ]
    global_pri = global_pri[ :, y_start:y_start+128, x_start:x_start+128 ]
    global_seg = global_seg[ :, y_start:y_start+128, x_start:x_start+128 ]
    crop_indicator = torch.tensor([0]) # if cropped then it will not be used in global D 

    # recalculate box location and size 
    x1, y1, x2, y2 = get_box( np.array(global_seg.squeeze()) )
    x2, y2 = modify_x2y2(x2, y2, args.prior_size, args.prior_size) 
    info = { 'new_box':[x1,y1,x2,y2],  'new_size':[y2-y1, x2-x1]  }

    return target_img, target_seg, global_img, global_sem, global_pri, global_seg, crop_indicator, info


def directly_resize(datum, args):

    target_img, target_seg, global_img, global_sem, global_pri, global_seg, crop_indicator, info = datum

    # no crop, we ditectly resize to 256 and 128, default is nearest    
    target_img = F.interpolate(  target_img.unsqueeze(0), size=(256,256)   ).squeeze(0)
    target_seg = F.interpolate(  target_seg.unsqueeze(0), size=(256,256)   ).squeeze(0)
    global_img = F.interpolate(  global_img.unsqueeze(0), size=(128,128)   ).squeeze(0)
    global_sem = F.interpolate(  global_sem.unsqueeze(0).float(), size=(128,128)   ).squeeze(0).long()
    global_pri = F.interpolate(  global_pri.unsqueeze(0), size=(128,128)   ).squeeze(0)
    global_seg = F.interpolate(  global_seg.unsqueeze(0), size=(128,128)   ).squeeze(0)
 
    # recalculate box location and size 
    x1, y1, x2, y2 = get_box( np.array(global_seg.squeeze()) )
    x2, y2 = modify_x2y2(x2, y2, args.prior_size, args.prior_size) 
    info = { 'new_box':[x1,y1,x2,y2],  'new_size':[y2-y1, x2-x1]  }

    return target_img, target_seg, global_img, global_sem, global_pri, global_seg, crop_indicator, info



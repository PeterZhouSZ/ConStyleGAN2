#from dataset.instance_dataset.process_funs import process as instance_process
from dataset.utils.functions import *
import torch 
import torchvision.transforms.functional as TF
from torchvision import transforms    

import random

# def get_foreground_data(args, img, sem, ins, parsing):    
#     output = []    
#     order = parsing['order']
#     order_name = parsing['order_name']
#     for ins_idx, class_name in zip(order, order_name):        
#         box = get_box( np.array(ins)==ins_idx )        
#         if exist_check(*box):  # see EXPLANTION_1 
#             not_shown_inss = order[order.index(ins_idx):]
#             out = to_dict( instance_process(args, img, sem, ins, ins_idx, box, not_shown_inss, mode='image')  )
#             out['class_name'] = class_name
#             output.append(out)
            
#     return output




# def to_dict(datum):
#     target_seg, global_sem, global_pri, global_seg, composition_seg, info  = datum     
#     out={}    
#     out['target_seg'] = target_seg.unsqueeze(0) # used in Foreground Generator
#     out['global_sem'] = global_sem.unsqueeze(0) # used in Encoder
#     out['global_pri'] = global_pri.unsqueeze(0) # used in Encoder
#     out['global_seg'] = global_seg.unsqueeze(0) # used in Encoder
#     out['composition_seg'] = composition_seg.unsqueeze(0) # used in composition 
#     out['info'] = info    
#     return out 



def get_scene_data(img, composition=None):
    """
    This function is used by both background_dataset and refiner_dataset

    composition is also PIL.Image, but it is a generated image
    """
    full_img = TF.to_tensor(img) #[0,1]
    c,h,w = full_img.shape

    # print('full img', full_img.shape)
    # data augmentation
    # color_jitter = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.5, hue=0.5), transforms.ToTensor()])
    # D = color_jitter(full_img[:,:,h:2*h])
    # gamma = 0.8+random.random()*0.4 # [0.8~1.2]
    # H = full_img[0:1,:,0:h]**gamma
    # R = full_img[0:1,:,2*h:3*h]**gamma

    # extract conditional mask
    # scene_img = torch.cat([H, D, R], dim=0)
    scene_img = torch.cat([full_img[0:1,:,0:h], full_img[:,:,h:2*h], full_img[0:1,:,2*h:3*h]], dim=0)
    scene_sem = full_img[0:1,:,3*h:4*h]

    # scene_sem = torch.tensor( np.array(sem) ).unsqueeze(0).long()
    # scene_ins = torch.tensor( np.array(ins) ).unsqueeze(0).long()
    
    # if composition != None:
    #     composition = ( TF.to_tensor(composition) - 0.5 ) / 0.5 

    out = {}
    out['scene_img'] = 2*scene_img-1
    out['scene_sem'] = 2*scene_sem-1
    # out['scene_ins'] = scene_ins
    # if composition != None:
    #     out['composition'] = composition
    return out




# def process(args, img, sem, ins, parsing):
#     """   

#     This is the entrance of processing used by composition_dataset.py

#     img: PIL.Image with the original shape 
#     sem: PIL.Image with the original shape
#     ins: PIL.Image with the original shape
#     parsing: a refined parsing (a dict)
    
#     It will return 
#     bg_data: a dict containing data needed for background generator 
#     fg_data: a list containing multiple dict which stores data needed for foreground generator   
#     """
    
#     bg_data = get_scene_data(img, sem, ins, parsing=parsing)    
#     fg_data = get_foreground_data(args, img, sem, ins, parsing)    
#     return bg_data, fg_data










################          EXPLANTION_1            ################## 
#
# remember that these those images are resized to scene_size
# thus potentially it may remove small instances, 
# while our parsing is derived from the orignal resolution.
#
#######################################################################
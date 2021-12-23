import numpy as np
import random
import os 

    
def modify_x2y2(x2, y2, width, height):
    "see bottom why we add 1 to x2 and y2"
    x2 = min(x2+1, width)
    y2 = min(y2+1, height)
    return x2, y2 
    

def exist_check(x1, y1, x2, y2):
    if x1 == y1 ==x2 == y2 == 0:
        return False 
    else:
        return True
    
        
def enlarge_box(x1, y1, x2, y2, width, height, ratio):
    w, h = x2-x1, y2-y1
    r = int( max(w,h) * (ratio/2) )
    center_x = int( (x1+x2)/2 )
    center_y = int( (y1+y2)/2 )
    y1 = max(0, center_y-r)
    y2 = min(height, center_y+r)
    x1 = max(0, center_x-r)
    x2 = min(width, center_x+r)
    return x1, y1, x2, y2



def get_box(mask):
    "mask should be a 2D np.array " 
    y,x = np.where(mask == 1)
    x1,x2,y1,y2 = x.min(),x.max(),y.min(),y.max()
    w = x2-x1
    h = y2-y1
    return x1,y1,x2,y2



        
        
def random_shift_box(x1, y1, x2, y2, width, height, ratio):
    # if ratio=0.2 then shift range is from -0.2*half_w to 0.2*half_w 
    half_w, half_h = int((x2-x1)/2), int((y2-y1)/2)
    center_x = int( (x1+x2)/2 ) + random.randint( int(-ratio*half_w), int(ratio*half_w)  )
    center_y = int( (y1+y2)/2 ) + random.randint( int(-ratio*half_h), int(ratio*half_h)  )

    y1 = max(0, center_y-half_h)
    y2 = min(height, center_y+half_h)
    x1 = max(0, center_x-half_w)
    x2 = min(width, center_x+half_w)
    return x1, y1, x2, y2





def check_and_return(path):
    assert os.path.exists(path)
    return path 



def get_path(names, class_name, train=True):
    """
    
    names should be a list contaiing paths to dataset
    
    All dataset folder should follow the same structure:
        full_data:
            images
            annotations
            annotations_instance 
            ...
        class1_info
        class2_info
        ...
    
    It will return a path_list used by dataset 
                
    """
    
    temp = 'training' if train else 'validation'
    
    output = []
    for name in names:
        paths = {} 

        paths['img'] = check_and_return( os.path.join(name,'full_data','images',temp)  )
        paths['sem'] = check_and_return( os.path.join(name,'full_data','annotations',temp ) )
        paths['ins'] = check_and_return( os.path.join(name,'full_data','annotations_instance',temp)  )
        paths['info'] = check_and_return( os.path.join(name, class_name,temp+'_info.txt')  )
        paths['parsing'] = check_and_return( os.path.join(name,'full_data','parsing_'+temp+'.txt')  )
        output.append(paths)
        
    return output 
        

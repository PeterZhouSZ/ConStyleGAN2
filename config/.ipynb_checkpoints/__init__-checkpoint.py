import time 

time.sleep(3)
print('PLEASE MAKE SURE YOUR SELECT FUNCTION IS CORRECT')



def select(boxes, pixels):
    """
    
    Please select data used in training here. Since in original dataset, each image 
    may have different resolution and each instance' resolution also varies a lot. If you
    want to train a 'bed' specific generator at 256 you probally do not want to use 'bed' 
    that are very small. So you need to choose what should be your training data.  

    The below implrmentation is based on two policies:

        if box_size is smaller than certain threshold then remove it
        if pixel/box_size is smaller certain threshold then remove it


    boxes: a list contains boxes for all instance
    pixels: a list contains number_of_pixels for all instance

    boxes:  [   [xyxy], [xyxy], [xyxy], ....   ]
    pixels: [     N,      N,      N,    ...    ]

    please return a list which contains the index for selected instances
    
    """

    need_idxs = []
    for i, (box, pixel) in enumerate( zip(boxes, pixels) ):
        
        x1,y1,x2,y2 = box 
        policy1 = (x2-x1+1)>32 and (y2-y1+1)>32
        policy2 = pixel / (x2-x1+1)*(y2-y1+1) > 0.1

        if policy1 and policy2 :
            need_idxs.append(i)

    return need_idxs
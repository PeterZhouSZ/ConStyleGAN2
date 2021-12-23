
"""

Config includes the following variables:


DATASET: a string 
    We do not support combining multiple datasets when do compostion, which means unlike in the training of
    scene_model and instance_model where you can combine multiple datasets in their config files, here 
    DATASET has to be a string to one dataset.


PRETRAINED_MODELS: a list containing multiple dicts
    Each dict is infomation for one specific class. It has least has one key and value. key is class name, 
    value is path to this pre trained class specific model. An option key is code_dependency_idx, its value 
    is a int. If multiple classes use the same code_dependency_idx, then durining generation, all instances 
    among these class will use the same z code. This is useful in some cases, such as when you try to improve
    both shoes in an images, then even though you instance model can see blured input, it is likely it will
    generate different shoes(though roughly color is same). If one class has the code_dependency_idx, then 
    at least one more model needs to have this. If you do not want this propert then simply do not specify 
    this. Also order of each dict in this list is important.It will generate each class instances according
    to this order. For example if you give [ {"bed": "path1"}, {"pillow": "path2"} ], then it will first 
    generate bed instances one by one and then composite on to base image, then the new image will be used
    as context to generate pillows. For the order within the same class, say you have two bed in this images, 
    the order is defined as instance_idx annotation in your dataset. TODO: I am not sure is this true?


PRETRAINED_BASE_MODEL: a string 
    In composite.py, if from_semantic is true then this must be given. See composite.py for details

"""







class CompositionConfig():


    DATASET = '/home/code-base/scratch_space/Server/data/human_data/human_standing/'
    
    
    PRETRAINED_MODELS = [
                            #{"face": "output/face/celeba_noncontext/checkpoint/130000.pt"},
                             #{"face": "output/face/visible/checkpoint/160000.pt"},
                             #{"face": "output/face_nocontext_nosem/checkpoint/020000.pt"},
                             #{"right_shoe": "output/shoes/visible/checkpoint/150000.pt",'code_dependency_idx':1},
                             #{"left_shoe": "output/shoes/visible/checkpoint/150000.pt", 'code_dependency_idx':1},
                            #{"upperclothes": "output/upperclothes/notvisible/checkpoint/200000.pt"}
                            
                       
                  
                            
                        ]  

    PRETRAINED_BASE_MODELS = '../VERSION/version5/output/person/checkpoint/300000.pt'



 
       
    
# class CompositionConfig():


#     DATASET = '/home/code-base/scratch_space/Server/data/cityscape_ood'
    
    
#     PRETRAINED_MODELS = [
#                             {"car": "output/car/novisible_extra/checkpoint/150000.pt"},
#                             {"person": "output/person/notvisible/checkpoint/050000.pt"},
#                             #{"rider": "output/noextra_rider/checkpoint/040000.pt"},
#                             #{"motorcycle": "output/noextra_motocycle/checkpoint/040000.pt"},
#                             #{"bicycle": "output/noextra_bicycle/checkpoint/040000.pt"},
        
                  
                            
#                         ]  
#     PRETRAINED_BASE_MODELS = 'output/city/checkpoint/066000.pt'




  
# class CompositionConfig():


#     DATASET = '/home/code-base/scratch_space/Server/data/bedroom_ood'
    
    
#     PRETRAINED_MODELS = [
#          #{"bed": "output/bed/visible/checkpoint/160000.pt"},
#          #{"chest of drawers, chest, bureau, dresser": "output/chest/visible/checkpoint/100000.pt"},
#          #{"pillow": "output/pillow/visible/checkpoint/140000.pt"},
#          #{"table": "output/table/visible/checkpoint/180000.pt"},
#          {"lamp": "output/lamp/notvisible/checkpoint/140000.pt"},
#          #{"chair": "output/chair/visible/checkpoint/180000.pt"},
       
                    
                  
                            
#                         ]  

#     PRETRAINED_BASE_MODELS = '../VERSION/version5/output/background/checkpoint/270000.pt'


















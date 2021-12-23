






class InstanceConfig():
    OBJ_PATH = '/home/code-base/scratch_space/Server/data/ade20k/objectInfo150.txt'

    TRAIN_DATASETS = [  
#                      {'dataset':'/home/code-base/scratch_space/Server/data/iMaterialist',
#                          'class':'bed'},
                       {'dataset':'/home/code-base/scratch_space/Server/data/ade_bedroom',
                         'class':'chest of drawers, chest, bureau, dresser'} ,
                       {'dataset':'/home/code-base/scratch_space/Server/data/places/bedroom',
                         'class':'chest of drawers, chest, bureau, dresser'},
                       {'dataset':'/home/code-base/scratch_space/Server/data/places/hotel_room',
                         'class':'chest of drawers, chest, bureau, dresser'},
#                       {'dataset':'/home/code-base/scratch_space/Server/data/places/childs_room',
#                          'class':'lamp'},
#                       {'dataset':'/home/code-base/scratch_space/Server/data/places/dining_room',
#                          'class':'lamp'},
#                       {'dataset':'/home/code-base/scratch_space/Server/data/places/living_room',
#                          'class':'lamp'},
   
                     ]

    
    TEST_DATASETS = [  {'dataset':'/home/code-base/scratch_space/Server/data/ade_bedroom',
                         'class':'chest of drawers, chest, bureau, dresser'}
                     ]









# class InstanceConfig():
#     OBJ_PATH = '/home/code-base/scratch_space/Server/data/human_data/human_standing/objectInfo.txt'

#     TRAIN_DATASETS = [  
# #                         {'dataset':'/home/code-base/scratch_space/Server/data/human_body/standing',
# #                          'class':'left_shoe'},
# #                         {'dataset':'/home/code-base/scratch_space/Server/data/human_body/standing',
# #                          'class':'right_shoe'},
# #                         {'dataset':'/home/code-base/scratch_space/Server/data/human_body/non_standing',
# #                          'class':'left_shoe'},
# #                         {'dataset':'/home/code-base/scratch_space/Server/data/human_body/non_standing',
# #                          'class':'right_shoe'},
#                         {'dataset':'/home/code-base/scratch_space/Server/data/Celeba',
#                          'class':'face'},
                      
#                      ]

    
#     TEST_DATASETS = [  
#                         {'dataset':'/home/code-base/scratch_space/Server/data/human_data/human_standing',
#                         'class':'face'},



#                     ]








# class InstanceConfig():
#     OBJ_PATH = '/home/code-base/scratch_space/Server/data/cityscape/objectInfo.txt'

#     TRAIN_DATASETS = [  
#                         {'dataset':'/home/code-base/scratch_space/Server/data/cityscape',
#                          'class':'rider'},
# #                         {'dataset':'/home/code-base/scratch_space/Server/data/cityscape_extra',
# #                          'class':'car'},
# #                             {'dataset':'/home/code-base/scratch_space/Server/data/caltech',
# #                             'class':'car'},
        
#                      ]
    
#     TEST_DATASETS = [  {'dataset':'/home/code-base/scratch_space/Server/data/cityscape',
#                          'class':'motorcycle'} # rider, bicycle
                     
#                      ]











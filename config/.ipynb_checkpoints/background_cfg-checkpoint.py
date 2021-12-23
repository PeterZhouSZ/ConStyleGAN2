from config.base_cfg import BaseConfig
from config.utils import check 

# class BackgroundConfig(BaseConfig):
#     TRAIN_DATASETS = ['/home/code-base/scratch_space/Server/data/ade_bedroom']
#                     #   '/home/code-base/scratch_space/Server/data/places/bedroom',
#                     #   '/home/code-base/scratch_space/Server/data/places/hotel_room'
#                     #   ]
#     TEST_DATASETS = ['/home/code-base/scratch_space/Server/data/ade_bedroom'] 

  


class BackgroundConfig(BaseConfig):
    TRAIN_DATASETS = ['/home/code-base/scratch_space/Server/data/human_data/human_standing']
    TEST_DATASETS = ['../../human_data/human_standing'] 

"""

Your data must be stored in the following way. annotations is the folder for semantic maps. 
annotations_instance is the folder for instance maps.
 

PATH_TO_DATASET:

    full_data

        images
            training
                1.jpg
                2.jpg
                ...
            validation
                1.jpg
                2.jpg
                ...
            
        annotations
            training
                1.png
                2.png
                ...
            validation
                1.png
                2.png
                ...

        annotations_instance
            training
                1.png
                2.png
                ...
            validation
                1.png
                2.png
                ... 



"""


class SceneConfig():
    TRAIN_DATASETS = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/BricksDataset_pat2_train']
    TEST_DATASETS = ['/home/grads/z/zhouxilong199213/Projects/Adobe_2021S/Dataset/BricksDataset_pat2_val']

    # TRAIN_DATASETS = ['/home/code-base/scratch_space/Server/data/ade_bedroom']
    # TEST_DATASETS = ['/home/code-base/scratch_space/Server/data/ade_bedroom'] 





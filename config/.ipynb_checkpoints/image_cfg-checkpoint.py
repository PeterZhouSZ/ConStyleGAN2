from config.base_cfg import ConfigBase
from config.utils import check 

class ImageConfig(ConfigBase):
    TRAIN_DATASETS = ['/home/code-base/scratch_space/Server/data/ade_bedroom']
    TEST_DATASETS = ['/home/code-base/scratch_space/Server/data/ade_bedroom']

    # It has to follow the order: foreground order specified in BaseConfig + background 
    PRETRAINED_MODELS = ['/home/code-base/scratch_space/Server/my_stylegan3/VERSION/version1/output/SCENEGAN_RELATED/nonconstant16vgg/bed/checkpoint',
                        '/home/code-base/scratch_space/Server/my_stylegan3/VERSION/version1/output/SCENEGAN_RELATED/nonconstant16vgg/cabinet/checkpoint',
                        '/home/code-base/scratch_space/Server/my_stylegan3/VERSION/version1/output/SCENEGAN_RELATED/nonconstant16vgg/chair/checkpoint',
                        '/home/code-base/scratch_space/Server/my_stylegan3/VERSION/version1/output/SCENEGAN_RELATED/nonconstant16vgg/chest/checkpoint',
                        '/home/code-base/scratch_space/Server/my_stylegan3/VERSION/version1/output/SCENEGAN_RELATED/nonconstant16vgg/cushion/checkpoint',
                        '/home/code-base/scratch_space/Server/my_stylegan3/VERSION/version1/output/SCENEGAN_RELATED/nonconstant16vgg/lamp/checkpoint',
                        '/home/code-base/scratch_space/Server/my_stylegan3/VERSION/version1/output/SCENEGAN_RELATED/nonconstant16vgg/pillow/checkpoint',
                        '/home/code-base/scratch_space/Server/my_stylegan3/VERSION/version1/output/SCENEGAN_RELATED/nonconstant16vgg/table/checkpoint',
                         '/home/code-base/scratch_space/Server/my_stylegan3/BackgroundGenerator.pth'
                        ]  



check(ImageConfig)


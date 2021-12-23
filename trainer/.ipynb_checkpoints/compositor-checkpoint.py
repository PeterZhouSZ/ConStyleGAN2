import torch 
from models.instance_model import Generator as FGGenerator 
from models.scene_model import Generator as BGGenerator 
import os 
from config.composition_cfg import CompositionConfig
from dataset.image_dataset.composition_dataset import Dataset as CompositionDataset
from dataset.image_dataset.scene_dataset import Dataset as SceneDataset
from dataset.instance_dataset.dataset import Dataset as InstanceDataset
import torch.nn as nn
from trainer.merger import Merger
from torchvision import  utils
import torch.nn.functional as F
from trainer.utils import ImageSaver, CodeDependency, SemanticMapVisualizer
import shutil 
import torchvision.transforms.functional as TF
from trainer.instance_trainer import process_data as process_instance_data
from trainer.scene_trainer import process_data as process_scene_data
from PIL import Image
import PIL
import numpy as np


class Compositor():
    def __init__(self, args, device):
        super().__init__()
        
        self.main_args = args 
        self.device = device
        
        self.prepare_pretrained_models()
        

        if self.main_args.from_semantic:
            # we need to use SceneDataset to generate our starting image from semantic input
            self.dataset = SceneDataset(self.base_model['args'], self.main_args.use_train, [CompositionConfig.DATASET] )
        else:
            # we just use CompositionDataset which simply output the starting image
            self.dataset = CompositionDataset(self.main_args) 
        
        self.dependency = CodeDependency(CompositionConfig)
        self.semantic_visualizer = SemanticMapVisualizer(151, True)
        self.merger = Merger(args, device, ignore=False)        
        self.prepare_output_folder()
        
        
        
    def prepare_output_folder(self):
        if os.path.exists(self.main_args.save_path):
            shutil.rmtree(self.main_args.save_path)
        os.mkdir(  self.main_args.save_path  )
        
        path = os.path.join(self.main_args.save_path, 'base_image')
        os.mkdir(  path  )
        self.base_image_saver = ImageSaver( path )
        
        path = os.path.join(self.main_args.save_path, 'final_image')
        os.mkdir(  path  )
        self.final_image_saver = ImageSaver( path )
        
        path = os.path.join(self.main_args.save_path, 'sem')
        os.mkdir(  path  )
        self.sem_saver = ImageSaver( path )
        
        path = os.path.join(self.main_args.save_path, 'ins')
        os.mkdir(  path  )
        self.ins_saver = ImageSaver( path )
        
        path = os.path.join(self.main_args.save_path, 'vis')
        os.mkdir(  path  )
        self.vis_saver = ImageSaver( path )
        
        path = os.path.join(self.main_args.save_path, 'real')
        os.mkdir(  path  )
        self.real_saver = ImageSaver( path )


        
    def prepare_pretrained_models(self):
        "here we load pretrained fg and bg model"
        self.models = {}        
        self.class_names = []

        for path_dict in CompositionConfig.PRETRAINED_MODELS:
            class_name = list(path_dict.keys())[0]
            self.class_names.append(class_name)
            path = self.find_path(  path_dict[class_name]  )
            print("load ckpt: ", path)
            ckpt = torch.load(path, map_location=lambda storage, loc: storage)

            generator = FGGenerator( ckpt['args'], self.device ).to(self.device)
            generator.load_state_dict(ckpt["g_ema"])
            self.models[class_name] = { 'model':generator.eval(), 'args':ckpt['args']  }

        # load base model if starting from semantic 
        if self.main_args.from_semantic:
            print("generation from semantic, load ckpt: ", CompositionConfig.PRETRAINED_BASE_MODELS)
            path = self.find_path(  CompositionConfig.PRETRAINED_BASE_MODELS  )
            ckpt = torch.load(path, map_location=lambda storage, loc: storage)

            # modify base_model args 
            ckpt['args'].random_flip = False
            ckpt['args'].random_crop = False

            # change scene_size in main_args base on base_model args
            self.main_args.scene_size = ckpt['args'].scene_size

            generator = BGGenerator( ckpt['args'], self.device ).to(self.device)
            generator.load_state_dict( ckpt["g_ema"]  )
            self.base_model = {  "model":generator.eval(), 'args':ckpt['args']  }

   

    def find_path(self, path):
        "If path is a file then return it, otherwise return file inside the folder"
        if os.path.isfile(path):
            return path 
        else:
            files = os.listdir(path)
            assert len(files) == 1, 'multiple files in the given folder ' + path
            return os.path.join( path,files[0] )
        
   
    def unsqueeze(self, input):
        for key in input:
            if type(input[key]) == torch.Tensor:
                input[key] = input[key].unsqueeze(0)
        return input 


    def convert(self, x):
        " it will convert PIL.Image/Tensor to Tensor/PIL.Image. Tensor (3*H*W) should/will have range of -1, 1"
        if type(x) == PIL.Image.Image:
            return ( TF.to_tensor(x) - 0.5 ) / 0.5
        if type(x) == torch.Tensor:
            assert x.ndim == 3
            # the following code of conversion is adapt from the Pytorch offical code: torchvision.utils.save_image
            def norm_ip(img, min, max):
                img = img.clamp(min=min, max=max)
                img = img.add(-min).div(max - min + 1e-5)
                return img
            x = norm_ip( x, -1, 1 )
            x = x.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            return Image.fromarray(x) 


    def modify_args(self,args):
        "input is used args"
        args.scene_size = self.main_args.scene_size # add a new argment which is used when do composition 
        args.random_flip = False
        return args


    def forward(self, generator, input_data, class_name):

        if self.dependency.check_dependency(class_name):
            code = self.dependency.get_code(class_name)
            if code == None:
                # this is the first instance, havent stored any code yet 
                with torch.no_grad():
                    output = generator( input_data['global_pri'], input_data['real_seg'], return_latents=True )
                generated_instance = output['image']
                latent = output['latent']
                self.dependency.store_code(class_name, latent)
            else:
                # then use this w+ code to generate instance
                with torch.no_grad():
                    output = generator( input_data['global_pri'], input_data['real_seg'], styles=code, input_type='w+' )
                generated_instance = output['image']

        else:
            # not dependent on any class, just do forwarding by itself
            with torch.no_grad():
                generated_instance = generator( input_data['global_pri'], input_data['real_seg'] )['image']

        return generated_instance


    
    
    def composite(self, current_image, global_idx ):

        for class_name in self.class_names:
          
            # get model and args used when train this generator
            generator = self.models[class_name]['model']
            instance_args = self.modify_args( self.models[class_name]['args'] )

            # create an instance_dataset for this Image 
            per_image = { 'dataset': [{'dataset':CompositionConfig.DATASET, 'class':class_name}], 
                          'wanted_image_idx':global_idx
                        }

            # Note instance_args is used when train class-specific generator, self.main_args is current script args
            instance_dataset = InstanceDataset( instance_args, self.main_args.use_train, per_image )
            total_instance = len(instance_dataset)

            for i in range(total_instance):

                # register current image and fetch data
                #current_image = current_image.resize( self.main_args.scene_size, Image.BICUBIC) #### This is eli's idea 
                instance_dataset.register(current_image) 
  
                data = self.unsqueeze( instance_dataset[i] )
                
                # process data 
                input_data = process_instance_data( instance_args, data, self.device )
                composition_seg = data['composition_seg'].to(self.device)
                
                # forward 
                generated_instance = self.forward( generator, input_data, class_name )
   
                
                # compsite     
                current_image = self.merger( generated_instance, 
                                             input_data['real_seg'], 
                                             self.convert(current_image).unsqueeze(0).to(self.device), 
                                             composition_seg, 
                                             [data['location_info']])
                current_image = self.convert(current_image.squeeze())
            
        return current_image


    def get_starting_image(self, i):
        "accoriding to if we directly load saved image or generate from semantic, we prepare starting image"
        if self.main_args.from_semantic:

            data = self.unsqueeze( self.dataset[i] ) 
            data = process_scene_data( self.base_model['args'], data, self.device  )
 
            with torch.no_grad():
                generator = self.base_model['model']
                image = generator(data['global_pri'])['image'].squeeze()
                
            
            # # # TEMP CODE # # # 
            real = Image.open( self.dataset.sourcer.img_bank[i]  ).resize((512,512) )
            sem = Image.open( self.dataset.sourcer.sem_bank[i]  ).resize((512,512), Image.NEAREST )
            ins = Image.open( self.dataset.sourcer.ins_bank[i]  ).resize((512,512), Image.NEAREST)
            data = self.unsqueeze( self.dataset[i] ) 
            vis = self.semantic_visualizer( data['scene_sem']  )
                        
            # # # # # # # # # # # 

                
                
            return self.convert(image), i, sem, ins, vis, real 
        else:
            return self.dataset[i]


        
    def start(self):    
        total_data = len(self.dataset)   
        for i in range( total_data ):
            print(i)
            
            if i in [4596,5615]:
                
                for j in range(10):
                    
                    starting_image, global_idx, sem, ins, vis, real  = self.get_starting_image( i )
                    #starting_image = Image.open('0330300011.png').resize((512,512))
       

                    composited_scene = self.composite( starting_image, global_idx )
                    self.base_image_saver(  starting_image, str(i).zfill(5)+str(j).zfill(5)+'.png' )
                    self.final_image_saver(  composited_scene, str(i).zfill(5)+str(j).zfill(5)+'.png' )
                    
                    self.sem_saver(  sem, str(i).zfill(5)+str(j).zfill(5)+'.png' )
                    self.ins_saver(  ins, str(i).zfill(5)+str(j).zfill(5)+'.png' )
                    self.vis_saver(  vis, str(i).zfill(5)+str(j).zfill(5)+'.png' )
                    self.real_saver(  real, str(i).zfill(5)+str(j).zfill(5)+'.png' )
                    
                    
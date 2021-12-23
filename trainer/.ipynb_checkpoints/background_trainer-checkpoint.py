import os
import sys
import shutil
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from tensorboardX import SummaryWriter
from copy import deepcopy

from models.scenegan_background import Generator, Discriminator
from dataset.image_dataset import get_background_dataloader
from misc.DiffAugment import DiffAugment

from trainer.utils import sample_data, ImageSaver
from trainer.merger import Merger
from criteria.gan import g_nonsaturating_loss, d_logistic_loss, g_path_regularize, d_r1_loss
from criteria.vgg import VGGLoss
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size 



def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def get_edges(t):
    ByteTensor = torch.cuda.ByteTensor
    edge = ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
    return edge.float()


def process_data(args, data, device):  

    scene_img, scene_sem, scene_ins, scene_seg = data

    if not args.consider_foreground:
        scene_img = scene_img*scene_seg

    real_img = scene_img.to(device)
    scene_sem = scene_sem.to(device)
    scene_ins = scene_ins.to(device)
    scene_seg = scene_seg.to(device)
    
    # shift by 1 if needed. 
    # In bg training we do not do zero padding on semantic, thus if no zero class in orginal annotation, we will never have 0 
    if not args.have_zero_class:
        scene_sem = scene_sem-1 

    # convert sem into channel representation  
    batch, _, height, width = scene_sem.shape
    all_zeros = torch.zeros( batch, args.number_of_semantic, height, width ).to(device)
    scene_sem = all_zeros.scatter_(1, scene_sem, 1.0)

    # concat with edge map 
    global_pri = torch.cat( [scene_sem, get_edges(scene_ins)], dim=1 )

    return real_img, global_pri, scene_seg



def sample_visualization_data(args, loader, device):

    sample_real_img = []
    sample_global_pri = []
    sample_scene_seg = []

    while True:
        real_img, global_pri, scene_seg = process_data(args, next(loader), device )    
        for i in range( args.batch_size ):   
            if len(sample_real_img) == args.n_sample: # break two loops here
                break
            sample_real_img.append( real_img[i] )
            sample_global_pri.append(  global_pri[i] )
            sample_scene_seg.append(  scene_seg[i] )    
        else:
            continue  
        break  
   
    output = {}
    output['real_img'] = torch.stack(sample_real_img)
    output['global_pri'] = torch.stack(sample_global_pri)
    output['scene_seg'] = torch.stack(sample_scene_seg)
    return output




class Trainer():
    def __init__(self, args, device):

        self.args = args
        self.device = device


        self.prepare_model()
        self.prepare_optimizer()
        self.prepare_dataloader()
        self.get_vgg_loss = VGGLoss()
        self.loss_dict = {}

        if self.args.ckpt:
            self.load_ckpt()  
        if self.args.distributed:
            self.wrap_module()





        if self.args.augment:
            augment_p = [float(s.strip()) for s in self.args.augment_p.split(',')]
            self.policy = [ ['cutout',augment_p[0]], ['color',augment_p[1]], ['translation',augment_p[2]] ]

        if get_rank() == 0:
            self.image_train_saver = ImageSaver(os.path.join('output',self.args.name,'sample_train'), int(self.args.n_sample**0.5) )
            self.image_test_saver = ImageSaver(os.path.join('output',self.args.name,'sample_test'), int(self.args.n_sample**0.5) )
            self.prepare_exp_folder()                
            self.prepare_visualization_data()                
        synchronize()
        
      
   

    def prepare_exp_folder(self):
        path = os.path.join( 'output', self.args.name  )
        shutil.rmtree(path) if os.path.exists(path) else os.makedirs(path)
        os.makedirs(path+'/checkpoint' )
        os.makedirs(path+'/sample_train' )
        os.makedirs(path+'/sample_test' )
        os.makedirs(path+'/Log' )    
        self.writer = SummaryWriter( path+'/Log' )
        shutil.copy2(sys.argv[0], path)


    def wrap_module(self):
        self.generator = DDP( self.generator, device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False )
        self.netD = DDP( self.netD, device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False )


    def prepare_model(self):
        self.generator = Generator(self.args,self.device).to(self.device)
        self.netD = Discriminator(self.args).to(self.device)
        self.g_ema = Generator(self.args,self.device).to(self.device)
        self.g_ema.eval()
        accumulate(self.g_ema, self.generator, 0)


    def prepare_optimizer(self):
        g_reg_ratio = self.args.g_reg_every / (self.args.g_reg_every + 1)    
        d_reg_ratio = self.args.d_reg_every / (self.args.d_reg_every + 1)  
        self.optimizerG = optim.Adam( self.generator.parameters(), lr=self.args.lr*g_reg_ratio, betas=(0**g_reg_ratio, 0.99**g_reg_ratio) )
        self.optimizerD = optim.Adam( self.netD.parameters(), lr=self.args.lr*d_reg_ratio, betas=(0**d_reg_ratio, 0.99**d_reg_ratio) )
     

    def load_ckpt(self):
       
        print("load ckpt: ", self.args.ckpt)
        ckpt = torch.load(self.args.ckpt, map_location=lambda storage, loc: storage)
        self.generator.load_state_dict(ckpt["generator"])
        self.netD.load_state_dict(ckpt["netD"])
        self.g_ema.load_state_dict(ckpt["g_ema"])

        self.optimizerG.load_state_dict(ckpt["optimizerG"])
        self.optimizerD.load_state_dict(ckpt["optimizerD"])

        try:
            self.args.start_iter = int( os.path.splitext(   os.path.basename(self.args.ckpt)   )[0] )
        except ValueError:
            pass
       


    def prepare_dataloader(self):
        self.train_loader = sample_data( get_background_dataloader(self.args, train=True)   )
        self.test_loader = sample_data( get_background_dataloader( self.args, train=False ) )


    def prepare_visualization_data(self):
        self.train_sample = sample_visualization_data(self.args, self.train_loader, self.device)
        self.image_train_saver( self.train_sample['real_img'], 'real.png' )      
        self.test_sample = sample_visualization_data(self.args, self.test_loader, self.device)
        self.image_test_saver( self.test_sample['real_img'], 'real.png' ) 
      

    def write_loss(self,count):
        for key in self.loss_dict:
            self.writer.add_scalar(  key, self.loss_dict[key], count  )


    def print_loss(self,count):
        print( str(count)+' iter finshed' )
        for key in self.loss_dict:
            print(key, self.loss_dict[key])
        print(' ')


    def visualize(self, count):
        with torch.no_grad():  
            output = self.g_ema( self.train_sample['global_pri'] )    
            self.image_train_saver( output['image'] , str(count).zfill(6)+'.png' )
            output = self.g_ema( self.test_sample['global_pri'] )     
            self.image_test_saver( output['image'] , str(count).zfill(6)+'.png' ) 
           

    def save_ckpt(self, count):
 
        save_dict =  {   "generator": self.g_module.state_dict(),
                         "netD": self.d_module.state_dict(),
                         "g_ema": self.g_ema.state_dict(),
                         "optimizerG": self.optimizerG.state_dict(),
                         "optimizerD": self.optimizerD.state_dict() }
        save_path = os.path.join( 'output', self.args.name, 'checkpoint', str(count).zfill(6)+'.pt' )
        torch.save(save_dict, save_path)

        old_ckpt = os.path.join( 'output', self.args.name, 'checkpoint', str(count-10000).zfill(6)+'.pt' )
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


    def trainD(self):

        if self.args.augment:
            augmented_real_img = DiffAugment( deepcopy(self.real_img), self.policy )
            augmented_fake_img = DiffAugment( deepcopy(self.fake_img.detach()), self.policy )
        else:
            augmented_real_img = self.real_img
            augmented_fake_img = self.fake_img.detach()

        real_pred = self.netD(augmented_real_img)
        fake_pred = self.netD(augmented_fake_img)        
        d_loss = d_logistic_loss(real_pred, fake_pred)  
        self.loss_dict["d"] = d_loss.item()

        self.optimizerD.zero_grad()
        d_loss.backward()
        self.optimizerD.step()


    def trainG(self):
       
        if self.args.augment:
            augmented_fake_img = DiffAugment( self.fake_img, self.policy )
        else:
            augmented_fake_img = self.fake_img

        fake_pred = self.netD(augmented_fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)
        self.loss_dict["g"] = g_loss.item()

        loss = g_loss + self.kl_loss

        self.optimizerG.zero_grad()
        loss.backward()
        self.optimizerG.step()


    def regularizeD(self):
       
        self.real_img.requires_grad = True
        real_pred = self.netD(self.real_img)

        r1_loss = d_r1_loss(real_pred, self.real_img)
        r1_loss = self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
        self.loss_dict["r1"] = r1_loss.item()

        self.optimizerD.zero_grad()
        r1_loss.backward()
        self.optimizerD.step()

       
    def regularizePath(self):

        output = self.generator(self.global_pri, return_latents=True, return_loss=False)
        fake_img = output['image']
        latents = output['latent']

        path_loss, self.mean_path_length = g_path_regularize( fake_img, latents, self.mean_path_length )
        path_loss = self.args.path_regularize * self.args.g_reg_every * path_loss + 0 * fake_img[0, 0, 0, 0]
        self.loss_dict['path_loss'] = path_loss.item()

        self.optimizerG.zero_grad()
        path_loss.backward()
        self.optimizerG.step()


    def regularizeVGG(self):
        randomize_noise = not self.args.vgg_fix_noise
        output = self.generator(self.global_pri, return_loss=False, randomize_noise=randomize_noise)
        fake_img = output['image']

        mask = None if self.args.consider_foreground else self.scene_seg
        vgg_loss = self.get_vgg_loss(fake_img, self.real_img, mask) * self.args.vgg_reg_every * self.args.vgg_regularize
        self.loss_dict['vgg_loss'] = vgg_loss.item()

        self.optimizerG.zero_grad()
        vgg_loss.backward()
        self.optimizerG.step()
   
   
    def train(self):
        "Note that in dist training printed and saved losses are not reduced, but from the first process"

        self.mean_path_length = 0

        if self.args.distributed:
            self.g_module = self.generator.module
            self.d_module = self.netD.module
        else:
            self.g_module = self.generator
            self.d_module = self.netD


        for idx in range(self.args.iter):
            count = idx + self.args.start_iter

            if count > self.args.iter:
                print("Done!")
                break

            self.real_img, self.global_pri, self.scene_seg  = process_data( self.args, next(self.train_loader), self.device )

            # forward G  
            output = self.generator(self.global_pri)
            self.fake_img = output['image']
            self.kl_loss = output['klloss']*self.args.kl_lambda
            self.loss_dict['kl'] = self.kl_loss.item()

            # update D
            self.trainD()
            if count % self.args.d_reg_every == 0:
                self.regularizeD()

            # update G
            self.trainG()
            if count % self.args.g_reg_every == 0:
                self.regularizePath()
            if self.args.vgg_reg_every != 0 and count % self.args.vgg_reg_every == 0:
                self.regularizeVGG()

            accum = 0.5 ** (32 / (10 * 1000))
            accumulate(self.g_ema, self.g_module, accum)

            if get_rank() == 0:
                if count % 10 == 0:
                    self.write_loss(count)
                if count % 50 == 0:
                    self.print_loss(count)
                if count % 500 == 0:
                    self.visualize(count)  
                if count % 10000 == 0:  
                    self.save_ckpt(count)
            synchronize()
import os
import sys
import shutil
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from tensorboardX import SummaryWriter
from copy import deepcopy

from models.instance_model import Generator, Discriminator
from dataset.instance_dataset import get_dataloader
from misc.DiffAugment import DiffAugment

from trainer.utils import sample_data, ImageSaver, to_device, sample_n_data, accumulate, blur, CheckpointSaver
from criteria.gan import g_nonsaturating_loss, d_logistic_loss, g_path_regularize, d_r1_loss
from criteria.vgg import VGGLoss
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size 
import time

# print('BG IS BLURED')

print('For context study, plase manually set rgb_prior=*0')


def process_data(args, data, device):  

    data = to_device(data, device)
    
    # This is for shoes class, we think it is better to blur the bg for both target_img and global_img 
    #data['target_img'] = blur( data['target_img'], 16,  1-data['target_seg']  )
    #data['global_img'] = blur( data['global_img'], 16,  1-data['global_seg']  )
    # This is for shoes class, we think it is better to blur the bg for both target_img and global_img 
    
    output = {}    
    output['real_img'] = data['target_img']
    output['real_seg'] = data['target_seg']

    # convert sem into channel representation  
    batch, _, height, width = data['global_sem'].shape
    channel = args.number_of_semantic if args.have_zero_class else args.number_of_semantic + 1
    all_zeros = torch.zeros( batch, channel, height, width ).to(device)
    data['global_sem'] = all_zeros.scatter_(1, data['global_sem'], 1.0)

    # prepare RBG prior 
    rgb_prior = data['global_img'] 
    if args.instance_visible and args.blur>0:
        rgb_prior = blur(rgb_prior, args.blur, data['global_seg'] )
    if not args.instance_visible:
        rgb_prior = rgb_prior*(1-data['global_seg'])
        
    
    #rgb_prior*=0   # no context at all (even no blur, everything is black)
    #rgb_prior = rgb_prior*data['global_seg']  # no context but instance is still there    
    


    # global_pri is the input to encoder
    output['global_pri'] = torch.cat( [rgb_prior, data['global_sem'], data['global_seg'] ], dim=1 )  # for nocontext nosem, just *0 for the first two
    

    return output




class Trainer():
    def __init__(self, args, device):

        self.args = args
        self.device = device

        self.prepare_model()
        self.prepare_optimizer()
        self.prepare_dataloader()
        self.loss_dict = {}
        self.get_vgg_loss = VGGLoss()

        if self.args.ckpt:
            self.load_ckpt()
        if self.args.distributed:
            self.wrap_module()

        if self.args.augment:
            augment_p = [float(s.strip()) for s in self.args.augment_p.split(',')]
            self.policy = [ ['cutout',augment_p[0]], ['color',augment_p[1]], ['translation',augment_p[2]] ]

        if get_rank() == 0:
            self.prepare_exp_folder()  
            self.image_train_saver = ImageSaver(os.path.join('output',self.args.name,'sample_train'), int(self.args.n_sample**0.5) )
            self.image_test_saver = ImageSaver(os.path.join('output',self.args.name,'sample_test'), int(self.args.n_sample**0.5) )
            self.ckpt_saver = CheckpointSaver( args, os.path.join('output',self.args.name,'checkpoint')  )
            self.writer = SummaryWriter( os.path.join('output',self.args.name,'Log') ) 
            self.prepare_visualization_data()
        synchronize()
        
   

    def prepare_exp_folder(self):
        path = os.path.join( 'output', self.args.name  )
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        os.makedirs(path+'/checkpoint' )
        os.makedirs(path+'/sample_train' )
        os.makedirs(path+'/sample_test' )
        os.makedirs(path+'/Log' )    
        shutil.copy2(sys.argv[0], path)


    def wrap_module(self):
        self.generator = DDP( self.generator, device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False )
        self.netD = DDP( self.netD, device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False )


    def prepare_model(self):
        self.generator = Generator(self.args,self.device).to(self.device)
        self.netD = Discriminator(self.args.data_size, self.args).to(self.device)
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

        self.args.start_iter = ckpt["iters"]
       


    def prepare_dataloader(self):
        self.train_loader = sample_data( self.args, get_dataloader(self.args, train=True)   )
        self.test_loader = sample_data(  self.args,  get_dataloader( self.args, train=False ) ) 


    def prepare_visualization_data(self):
        
        data = sample_n_data( self.args.n_sample, self.train_loader, self.args.batch_size )
        self.train_sample = process_data(self.args, data, self.device)
        self.image_train_saver( self.train_sample['real_img'], 'real.png' ) 

        data = sample_n_data( self.args.n_sample, self.test_loader, self.args.batch_size )
        self.test_sample = process_data(self.args, data, self.device)
        self.image_test_saver( self.test_sample['real_img'], 'real.png' ) 

      

    def write_loss(self,count):
        for key in self.loss_dict:
            self.writer.add_scalar(  key, self.loss_dict[key], count  )


    def print_loss(self,count):
        print( str(count)+' iter finshed' )
        for key in self.loss_dict:
            print(key, self.loss_dict[key])
        print('time has been spent in seconds since you lunched this script ', time.time()-self.tic)
        print(' ')


    def visualize(self, count):
        with torch.no_grad():  
            output = self.g_ema( self.train_sample['global_pri'], self.train_sample['real_seg'] )    
            self.image_train_saver( output['image'] , str(count).zfill(6)+'.png' )
            output = self.g_ema( self.test_sample['global_pri'], self.test_sample['real_seg'] )     
            self.image_test_saver( output['image'] , str(count).zfill(6)+'.png' ) 
           

    def save_ckpt(self, count):

        save_dict =  {   "args": self.args,
                         "generator": self.g_module.state_dict(),
                         "netD": self.d_module.state_dict(),
                         "g_ema": self.g_ema.state_dict(),
                         "optimizerG": self.optimizerG.state_dict(),
                         "optimizerD": self.optimizerD.state_dict(),
                         "iters": count }
        self.ckpt_saver( save_dict, count )



    def trainD(self):

        if self.args.augment:
            augmented_real_img = DiffAugment( deepcopy(self.data['real_img']), self.policy )
            augmented_fake_img = DiffAugment( deepcopy(self.fake_img.detach()), self.policy )
        else:
            augmented_real_img = self.data['real_img']
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
       
        self.data['real_img'].requires_grad = True
        real_pred = self.netD(self.data['real_img'])

        r1_loss = d_r1_loss(real_pred, self.data['real_img'])
        r1_loss = self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
        self.loss_dict["r1"] = r1_loss.item()

        self.optimizerD.zero_grad()
        r1_loss.backward()
        self.optimizerD.step()

       
    def regularizePath(self):

        output = self.generator(self.data['global_pri'], self.data['real_seg'], return_latents=True, return_loss=False)
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
        output = self.generator(self.data['global_pri'], self.data['real_seg'], return_loss=False, randomize_noise=randomize_noise)
        fake_img = output['image']
        target = self.data['real_img'] 
        if self.args.blur>0:
            target = blur( target, self.args.blur )
            fake_img = blur( fake_img, self.args.blur )
        vgg_loss = self.get_vgg_loss(fake_img, target) * self.args.vgg_reg_every * self.args.vgg_regularize
        self.loss_dict['vgg_loss'] = vgg_loss.item()

        self.optimizerG.zero_grad()
        vgg_loss.backward()
        self.optimizerG.step()
   
   
    def train(self):
        "Note that in dist training printed and saved losses are not reduced, but from the first process"

        self.mean_path_length = 0
        self.tic = time.time()

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

            self.data = process_data( self.args, next(self.train_loader), self.device )

            # forward G  
            output = self.generator( self.data['global_pri'], self.data['real_seg'] )
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
                if count % self.args.ckpt_save_frenquency == 0: 
                    self.save_ckpt(count)
            synchronize()
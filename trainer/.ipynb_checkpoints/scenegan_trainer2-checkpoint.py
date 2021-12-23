import os
import sys 
import shutil
import torch
from torch import optim
from torchvision import  utils
from tensorboardX import SummaryWriter
from copy import deepcopy

from models.scenegan2 import Generator, Discriminator
from dataset.instance_dataset import get_dataloader
from misc.DiffAugment import DiffAugment

from trainer.utils import sample_data
from trainer.merger import Merger
from criteria.gan import g_nonsaturating_loss, d_logistic_loss, g_path_regularize, d_r1_loss
from criteria.vgg import VGGLoss


"This is multiply mask one (The one we ignore bg for foreground object generation)"

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)



def process_data(data, device):   
    real_img, real_seg, global_img, global_sem, global_pri, global_seg, info = data

    real_img = real_img.to(device)
    real_seg = real_seg.to(device)
    global_img = global_img.to(device)
    global_sem = global_sem.to(device)
    global_pri = global_pri.to(device)
    global_seg = global_seg.to(device)

    # convert sem into channel representation  
    batch, _, height, width = global_sem.shape
    all_zeros = torch.zeros( batch, 151, height, width ).to(device)
    global_sem = all_zeros.scatter_(1, global_sem, 1.0)

    # concat global sem and global pri (here actually partially visible rgb)
    global_pri = torch.cat( [global_pri, global_sem, global_seg], dim=1 )

    return real_img*real_seg, real_seg, global_img, global_pri, global_seg, info



def sample_visualization_data(args, loader, device):

    sample_real_img = []
    sample_real_seg = []
    sample_global_img = []
    sample_global_pri = []
    sample_global_seg = []
    sample_info = []


    while True:
        real_img, real_seg, global_img, global_pri, global_seg, info = process_data( next(loader), device )    
        for i in range( args.batch_size ):
   
            if len(sample_real_img) == args.n_sample: # break two loops here
                break

            sample_real_img.append( real_img[i] )
            sample_real_seg.append( real_seg[i] )
            sample_global_img.append(  global_img[i]  )
            sample_global_pri.append(  global_pri[i] )
            sample_global_seg.append(  global_seg[i] )
            sample_info.append( info[i] )
            
        else:
            continue  
        break  
   
    output = {}
    output['real_img'] = torch.stack(sample_real_img)
    output['real_seg'] = torch.stack(sample_real_seg)
    output['global_img'] = torch.stack(sample_global_img)
    output['global_pri'] = torch.stack(sample_global_pri)
    output['global_seg'] = torch.stack(sample_global_seg)
    output['info'] = sample_info

    return output




class Trainer():
    def __init__(self, args, device):

        self.args = args 
        self.device = device

        self.prepare_model()
        self.prepare_optimizer()
        if self.args.ckpt:
            self.load_ckpt()

        if self.args.augment:
            augment_p = [float(s.strip()) for s in self.args.augment_p.split(',')]
            self.policy = [ ['cutout',augment_p[0]], ['color',augment_p[1]], ['translation',augment_p[2]] ]

            
        self.prepare_exp_folder()        
        self.prepare_dataloader() 
        self.prepare_visualization_data()
        
        self.loss_dict = {}
        self.get_vgg_loss = VGGLoss()
    

    def prepare_exp_folder(self):
        path = os.path.join( 'output', self.args.name  )
        shutil.rmtree(path) if os.path.exists(path) else os.makedirs(path)
        os.makedirs(path+'/checkpoint' )
        os.makedirs(path+'/sample_train' )
        os.makedirs(path+'/sample_test' )
        os.makedirs(path+'/Log' )    
        self.writer = SummaryWriter( path+'/Log' )
        shutil.copy2(sys.argv[0], path)


    def prepare_model(self):
        self.generator = Generator(self.args).to(self.device)
        self.netD_local = Discriminator(self.args.data_size).to(self.device) 
        self.g_ema = Generator(self.args).to(self.device)
        self.g_ema.eval()
        accumulate(self.g_ema, self.generator, 0)


    def prepare_optimizer(self):
        g_reg_ratio = self.args.g_reg_every / (self.args.g_reg_every + 1)    
        d_reg_ratio = self.args.d_reg_every / (self.args.d_reg_every + 1)   
        self.optimizerG = optim.Adam( self.generator.parameters(), lr=self.args.lr*g_reg_ratio, betas=(0**g_reg_ratio, 0.99**g_reg_ratio) )
        self.optimizerD_local = optim.Adam( self.netD_local.parameters(), lr=self.args.lr*d_reg_ratio, betas=(0**d_reg_ratio, 0.99**d_reg_ratio) )
     

    def load_ckpt(self):
        
        print("load ckpt: ", self.args.ckpt)
        ckpt = torch.load(self.args.ckpt, map_location=lambda storage, loc: storage)
        self.generator.load_state_dict(ckpt["generator"])
        self.netD_local.load_state_dict(ckpt["netD_local"])
        self.g_ema.load_state_dict(ckpt["g_ema"])

        self.optimizerG.load_state_dict(ckpt["optimizerG"])
        self.optimizerD_local.load_state_dict(ckpt["optimizerD_local"])

        try:
            self.args.start_iter = int( os.path.splitext(   os.path.basename(self.args.ckpt)   )[0] )
        except ValueError:
            pass
        


    def prepare_dataloader(self):
        self.train_loader = sample_data( get_dataloader(self.args, train=True)   )
        self.test_loader = sample_data( get_dataloader( self.args, train=False ) )


    def prepare_visualization_data(self):
        self.train_sample = sample_visualization_data(self.args, self.train_loader, self.device)
        save_path = os.path.join( 'output', self.args.name, 'sample_train', 'real.png' )
        utils.save_image( self.train_sample['real_img'], save_path, nrow=int(self.args.n_sample**0.5), normalize=True, range=(-1,1) )
     
        self.test_sample = sample_visualization_data(self.args, self.test_loader, self.device)
        save_path = os.path.join( 'output', self.args.name, 'sample_test', 'real.png' )
        utils.save_image( self.test_sample['real_img'], save_path, nrow=int(self.args.n_sample**0.5), normalize=True, range=(-1,1) )
       

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
            output = self.g_ema( self.train_sample['global_pri'], self.train_sample['real_seg'] )     
            save_path = os.path.join( 'output', self.args.name, 'sample_train', str(count).zfill(6)+'.png' )
            utils.save_image( output['image'], save_path, nrow=int(self.args.n_sample**0.5), normalize=True,range=(-1,1) )
           
            output = self.g_ema( self.test_sample['global_pri'], self.test_sample['real_seg'] )      
            save_path = os.path.join( 'output', self.args.name, 'sample_test', str(count).zfill(6)+'.png' )
            utils.save_image( output['image'], save_path, nrow=int(self.args.n_sample**0.5), normalize=True,range=(-1,1) )
         

    def save_ckpt(self, count):
 
        save_dict =  {   "generator": self.generator.state_dict(),
                         "netD_local": self.netD_local.state_dict(),
                         "g_ema": self.g_ema.state_dict(),
                         "optimizerG": self.optimizerG.state_dict(),
                         "optimizerD_local": self.optimizerD_local.state_dict() }
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

        real_pred = self.netD_local(augmented_real_img)
        fake_pred = self.netD_local(augmented_fake_img)        
        d_loss = d_logistic_loss(real_pred, fake_pred)  
        self.loss_dict["d"] = d_loss.item()

        self.netD_local.zero_grad()
        d_loss.backward() 
        self.optimizerD_local.step()


    def trainG(self):
        
        if self.args.augment:
            augmented_fake_img = DiffAugment( self.fake_img, self.policy )
        else:
            augmented_fake_img = self.fake_img

        fake_pred = self.netD_local(augmented_fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)
        self.loss_dict["g"] = g_loss.item()

        loss = g_loss + self.kl_loss

        self.optimizerG.zero_grad()
        loss.backward()
        self.optimizerG.step()


    def regularizeD(self):
        
        self.real_img.requires_grad = True
        real_pred = self.netD_local(self.real_img)

        r1_loss = d_r1_loss(real_pred, self.real_img)
        r1_loss = self.args.r1 / 2 * r1_loss * self.args.d_reg_every
        self.loss_dict["r1"] = r1_loss.item()

        self.netD_local.zero_grad()
        r1_loss.backward()
        self.optimizerD_local.step()

        
    def regularizePath(self):

        output = self.generator(self.global_pri, self.real_seg, return_latents=True, return_loss=False)
        fake_img = output['image']
        latents = output['latent']

        path_loss, self.mean_path_length = g_path_regularize( fake_img, latents, self.mean_path_length )
        path_loss = self.args.path_regularize * self.args.g_reg_every * path_loss
        self.loss_dict['path_loss'] = path_loss.item()

        self.optimizerG.zero_grad()
        path_loss.backward()
        self.optimizerG.step()


    def regularizeVGG(self):
        randomize_noise = not self.args.vgg_fix_noise
        output = self.generator(self.global_pri, self.real_seg, return_loss=False, randomize_noise=randomize_noise)
        fake_img = output['image']

        vgg_loss = self.get_vgg_loss(fake_img, self.real_img) * self.args.vgg_reg_every * self.args.vgg_regularize
        self.loss_dict['vgg_loss'] = vgg_loss.item()

        self.optimizerG.zero_grad()
        vgg_loss.backward()
        self.optimizerG.step()
    
    
    def train(self): 

        self.mean_path_length = 0

        for idx in range(self.args.iter):
            count = idx + self.args.start_iter

            if count > self.args.iter:
                print("Done!")
                break

            self.real_img, self.real_seg, self.global_img, self.global_pri, self.global_seg, self.info = process_data( next(self.train_loader), self.device )

            # forward G   
            output = self.generator(self.global_pri, self.real_seg)
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
            accumulate(self.g_ema, self.generator, accum)


            if count % 10 == 0:
                self.write_loss(count)
            if count % 50 == 0:
                self.print_loss(count)
            if count % 100 == 0:
                self.visualize(count)   
            if count % 10000 == 0:  
                self.save_ckpt(count)







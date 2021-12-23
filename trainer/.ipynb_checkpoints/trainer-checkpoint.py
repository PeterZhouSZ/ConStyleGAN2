import argparse
import os
import shutil
import torch
from torch import nn,  optim
from torchvision import  utils
from tensorboardX import SummaryWriter
from copy import deepcopy

from models.stylegan2.stylegan2 import Generator, Discriminator
from models.encoders.psp_encoders import GradualStyleEncoder
from dataset.instance_dataset import get_dataloader
from misc.DiffAugment import DiffAugment

from trainer.utils import sample_data, save_all_files
from trainer.ranger import Ranger
from trainer.merger import Merger
from criteria import w_norm
from criteria.lpips.lpips import LPIPS
from criteria.gan import g_nonsaturating_loss, d_logistic_loss







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

    return real_img, real_seg, global_img, global_pri, global_seg, info



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

        # prepare models 
        self.generator = Generator(args.data_size, args.style_dim, args.n_mlp).to(device)
        self.encoder = GradualStyleEncoder( input_nc=3+151+1 ).to(device) 
        if self.args.adv_training:
            self.netD_local = Discriminator(args.data_size).to(device) 
            self.netD_global = Discriminator(args.prior_size).to(device)

        self.load_ckpt()
        print('generator is frozen')
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False


        # prepare misc
        if self.args.adv_training:
            self.merger = Merger(device)
            if self.args.augment:
                augment_p = [float(s.strip()) for s in self.args.augment_p.split(',')]
                self.policy = [ ['cutout',augment_p[0]], ['color',augment_p[1]], ['translation',augment_p[2]] ]

            
        self.prepare_exp_folder()
        self.prepare_optimizer()
        self.prepare_dataloader() 
        self.prepare_visualization_data()
        

        self.loss_dict = {}
        self.mean_latent = self.generator.mean_latent(2048)

        if self.args.w_norm_lambda>0:
            self.get_w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=True)
        if self.args.l2_lambda>0:
            self.get_l2_loss = nn.MSELoss()
        if self.args.lpips_lambda>0:        
            self.get_lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.args.adv_training:
            self.g_nonsaturating_loss = g_nonsaturating_loss
            self.d_logistic_loss = d_logistic_loss

    def prepare_exp_folder(self):
        path = os.path.join( 'output', self.args.name  )
        shutil.rmtree(path) if os.path.exists(path) else os.makedirs(path)
        os.makedirs(path+'/checkpoint' )
        os.makedirs(path+'/sample_train' )
        os.makedirs(path+'/sample_test' )
        os.makedirs(path+'/Log' )    
        self.writer = SummaryWriter( path+'/Log' )
        save_all_files(path)


    def load_ckpt(self):

        if  self.args.ckpt:
            print("load both generator and encoder from the ckpt:", self.args.ckpt)
            ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(ckpt["g_ema"])
            self.encoder.load_state_dict(ckpt["e"])
            if self.args.adv_training:
                print("load both local and global discriminator from the ckpt:", self.args.ckpt)
                self.netD_local.load_state_dict(ckpt["d_local"])
                self.netD_global.load_state_dict(ckpt["d_global"])

        elif self.args.pretrained_G:
            ckpt = torch.load(self.args.pretrained_G, map_location=lambda storage, loc: storage)
            print("load pretrained generator from :", self.args.pretrained_G)            
            self.generator.load_state_dict(ckpt["g_ema"])
            if self.args.adv_training:
                #print("load local discriminator from the ckpt:", self.args.pretrained_G)
                #self.netD_local.load_state_dict(ckpt["d"])
                pass

        else:
            assert False, 'either ckpt or pretrained generator must be given'


    def prepare_optimizer(self):
        self.optimizerE = optim.Adam( self.encoder.parameters(), lr=self.args.lr )
        if self.args.adv_training:
            self.optimizerD_local = optim.Adam( self.netD_local.parameters(), lr=self.args.lr, betas=(0, 0.99) )
            self.optimizerD_global = optim.Adam( self.netD_global.parameters(), lr=self.args.lr, betas=(0, 0.99) )
            


    def prepare_dataloader(self):
        self.train_loader = sample_data( get_dataloader(self.args,train=True) )
        self.test_loader = sample_data( get_dataloader( self.args, train=False ) )


    def prepare_visualization_data(self):
        self.train_sample = sample_visualization_data(self.args, self.train_loader, self.device)
        save_path = os.path.join( 'output', self.args.name, 'sample_train', 'real.png' )
        utils.save_image( self.train_sample['real_img'], save_path, nrow=int(self.args.n_sample**0.5), normalize=True, range=(-1,1) )
        save_path = os.path.join( 'output', self.args.name, 'sample_train', 'real_global.png' )
        utils.save_image( self.train_sample['global_img'], save_path, nrow=int(self.args.n_sample**0.5), normalize=True, range=(-1, 1) )

        self.test_sample = sample_visualization_data(self.args, self.test_loader, self.device)
        save_path = os.path.join( 'output', self.args.name, 'sample_test', 'real.png' )
        utils.save_image( self.test_sample['real_img'], save_path, nrow=int(self.args.n_sample**0.5), normalize=True, range=(-1,1) )
        save_path = os.path.join( 'output', self.args.name, 'sample_test',  'real_global.png' )
        utils.save_image( self.test_sample['global_img'], save_path, nrow=int(self.args.n_sample**0.5), normalize=True, range=(-1, 1) )


    def write_loss(self,count):
        for key in self.loss_dict:
            self.writer.add_scalar(  key, self.loss_dict[key], count  )


    def print_loss(self,count):
        print( str(count)+' iter finshed' )
        for key in self.loss_dict:
            print(key, self.loss_dict[key])
        print(' ')


    def visualize(self, count):
        self.encoder.eval()
        with torch.no_grad():   
            w_plus_latent = self.encoder( self.train_sample['global_pri'] ) + self.mean_latent
            fake_img, _ = self.generator(w_plus_latent, input_type='w+')              
            save_path = os.path.join( 'output', self.args.name, 'sample_train', str(count).zfill(6)+'.png' )
            utils.save_image( fake_img, save_path, nrow=int(self.args.n_sample**0.5), normalize=True,range=(-1,1) )
            if self.args.adv_training:
                fake_global = self.merger(fake_img, self.train_sample['real_seg'], self.train_sample['global_img'], self.train_sample['global_seg'], self.train_sample['info'])
                save_path = os.path.join( 'output', self.args.name, 'sample_train', str(count).zfill(6)+'global.png' )
                utils.save_image( fake_global, save_path, nrow=int(self.args.n_sample**0.5), normalize=True, range=(-1,1) )


            w_plus_latent = self.encoder( self.test_sample['global_pri'] ) + self.mean_latent
            fake_img, _ = self.generator(w_plus_latent, input_type='w+')    
            save_path = os.path.join( 'output', self.args.name, 'sample_test', str(count).zfill(6)+'.png' )
            utils.save_image( fake_img, save_path, nrow=int(self.args.n_sample**0.5), normalize=True,range=(-1,1) )
            if self.args.adv_training:
                fake_global = self.merger(fake_img, self.test_sample['real_seg'], self.test_sample['global_img'], self.test_sample['global_seg'], self.test_sample['info'])
                save_path = os.path.join( 'output', self.args.name, 'sample_test', str(count).zfill(6)+'global.png' )
                utils.save_image( fake_global, save_path, nrow=int(self.args.n_sample**0.5), normalize=True,range=(-1,1) )
        
        self.encoder.train()


    def save_ckpt(self, count):
        # TODO: now only save encoder, which is not compitible with self.load_ckpt code
        save_dict = {"e": self.encoder.state_dict(),"args": self.args}
        save_path = os.path.join( 'output', self.args.name, 'checkpoint', str(count).zfill(6)+'.pt' )
        torch.save(save_dict, save_path)
        old_ckpt = os.path.join( 'output', self.args.name, 'checkpoint', str(count-10000).zfill(6)+'.pt' )
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


    def cal_E_non_adv_loss(self):
        loss = 0 
        if self.args.w_norm_lambda>0:
            w_norm_loss = self.get_w_norm_loss(self.w_plus_latent, self.mean_latent)*self.args.w_norm_lambda
            loss += w_norm_loss
            self.loss_dict['w_norm_loss'] = w_norm_loss.item()
        if self.args.l2_lambda>0:
            l2_loss = self.get_l2_loss( self.fake_img, self.real_img )*self.args.l2_lambda
            loss += l2_loss
            self.loss_dict['l2_loss'] = l2_loss.item()
        if self.args.lpips_lambda>0:        
            lpips_loss = self.get_lpips_loss( self.fake_img, self.real_img )*self.args.lpips_lambda
            loss += lpips_loss
            self.loss_dict['lpips_loss'] = lpips_loss.item()
        return loss


    def trainD(self):

        if self.args.augment:
            augmented_real_img = DiffAugment( deepcopy(self.real_img), self.policy )
            augmented_fake_img = DiffAugment( deepcopy(self.fake_img.detach()), self.policy )
            augmented_real_global = DiffAugment( deepcopy(self.global_img), self.policy )
            augmented_fake_global = DiffAugment( deepcopy(self.fake_global.detach()), self.policy )
        else:
            augmented_real_img = self.real_img
            augmented_fake_img = self.fake_img.detach()
            augmented_real_global = self.global_img
            augmented_fake_global = self.fake_global.detach()

        real_pred = self.netD_local(augmented_real_img)
        fake_pred = self.netD_local(augmented_fake_img)        
        d_loss = self.d_logistic_loss(real_pred, fake_pred)  
        self.loss_dict["d"] = d_loss.item()

        real_pred = self.netD_global(augmented_real_global)
        fake_pred = self.netD_global(augmented_fake_global)        
        d_global_loss = d_logistic_loss(real_pred, fake_pred)        
        self.loss_dict["d_global"] = d_global_loss.item()

        self.netD_local.zero_grad()
        self.netD_global.zero_grad()
        (d_loss+d_global_loss).backward()  # these two losses are from two completely different models
        self.optimizerD_local.step()
        self.optimizerD_global.step()



    def trainE(self):
        
        loss = self.cal_E_non_adv_loss()

        if self.args.adv_training:

            if self.args.augment:
                augmented_fake_img = DiffAugment( self.fake_img, self.policy )
                augmented_fake_global = DiffAugment( self.fake_global, self.policy )
            else:
                augmented_fake_img = self.fake_img
                augmented_fake_global = self.fake_global

            fake_pred = self.netD_local(augmented_fake_img)
            g_loss = self.g_nonsaturating_loss(fake_pred)
            self.loss_dict["g"] = g_loss.item()

            fake_pred = self.netD_global(augmented_fake_global)
            g_global_loss = self.g_nonsaturating_loss(fake_pred)
            self.loss_dict["g_global"] = g_global_loss.item()

            loss = loss + g_loss + g_global_loss

        self.optimizerE.zero_grad()
        loss.backward()
        self.optimizerE.step()
    
    
    def train(self): 

        for idx in range(self.args.iter):
            count = idx + self.args.start_iter

            if count > self.args.iter:
                print("Done!")
                break

            self.real_img, self.real_seg, self.global_img, self.global_pri, self.global_seg, self.info = process_data( next(self.train_loader), self.device )

            # get w+ code and forward G
            self.w_plus_latent = self.encoder(self.global_pri) # bs*14*512
            self.w_plus_latent += self.mean_latent
            self.fake_img, _ = self.generator(self.w_plus_latent, input_type='w+')

            if self.args.adv_training:
                self.fake_global = self.merger(self.fake_img, self.real_seg, self.global_img, self.global_seg, self.info)
               

            # update 
            if self.args.adv_training:
                self.trainD()
            self.trainE()

            
            if count % 10 == 0:
                self.write_loss(count)
            if count % 50 == 0:
                self.print_loss(count)
            if count % 100 == 0:
                self.visualize(count)   
            if count % 10000 == 0:  
                self.save_ckpt(count)







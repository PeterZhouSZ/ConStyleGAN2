import argparse
import os
import shutil
import torch
from torch import nn,  optim
from torchvision import  utils
from tensorboardX import SummaryWriter


from models.stylegan2.stylegan2 import Generator
from models.encoders.psp_encoders import GradualStyleEncoder
from dataset.celeba_dataset import get_dataloader


from trainer.utils import sample_data, save_all_files
from trainer.ranger import Ranger
from criteria import w_norm
from criteria.lpips.lpips import LPIPS


"This is just to verify if my code is correct by runing celeba"





def process_data(data, device):   
    real_img, real_sem = data

    real_img = real_img.to(device)
    real_sem = real_sem.to(device)
    
    
    batch, _, height, width = real_sem.shape
    all_zeros = torch.zeros( batch, 19, height, width ).to(device)
    real_sem = all_zeros.scatter_(1, real_sem, 1.0)
    
    return real_img, real_sem



def sample_visualization_data(args, loader, device):

    sample_real_img = []
    sample_real_seg = []
    sample_global_img = []
    sample_global_pri = []
    sample_global_seg = []
    sample_info = []


    while True:
        real_img, real_seg = process_data( next(loader), device )    
        for i in range( args.batch_size ):
   
            if len(sample_real_img) == args.n_sample: # break two loops here
                break

            sample_real_img.append( real_img[i] )
            sample_real_seg.append( real_seg[i] )
        else:
            continue  
        break  
   
    output = {}
    output['real_img'] = torch.stack(sample_real_img)
    output['real_seg'] = torch.stack(sample_real_seg)

    return output




class Trainer():
    def __init__(self, args, device):
        
        print('This is to train celeba, please manually make num_w = 18 in encoder!!')
        import time
        time.sleep(3)

        self.args = args 
        self.device = device

        self.generator = Generator(args.data_size, args.style_dim, args.n_mlp).to(device)
        self.encoder = GradualStyleEncoder( input_nc=19 ).to(device) 

        self.load_ckpt()

        print('generator is frozen')
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False
            
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
        elif self.args.pretrained_G:
            print("load pretrained generator from :", self.args.pretrained_G)
            ckpt = torch.load(self.args.pretrained_G, map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(ckpt["g_ema"])
        else:
            assert False, 'either ckpt or pretrained generator must be given'


    def prepare_optimizer(self):
        self.optimizerE = Ranger( self.encoder.parameters(), lr=self.args.lr )


    def prepare_dataloader(self):
        self.train_loader = sample_data( get_dataloader(self.args,train=True) )
        self.test_loader = sample_data( get_dataloader( self.args, train=False ) )


    def prepare_visualization_data(self):
        self.train_sample = sample_visualization_data(self.args, self.train_loader, self.device)
        save_path = os.path.join( 'output', self.args.name, 'sample_train', 'real.png' )
        utils.save_image( self.train_sample['real_img'], save_path, nrow=int(self.args.n_sample**0.5), normalize=True )
    
        self.test_sample = sample_visualization_data(self.args, self.test_loader, self.device)
        save_path = os.path.join( 'output', self.args.name, 'sample_test', 'real.png' )
        utils.save_image( self.test_sample['real_img'], save_path, nrow=int(self.args.n_sample**0.5), normalize=True )
    

    def write_loss(self,count):
        for key in self.loss_dict:
            self.writer.add_scalar(  key, self.loss_dict[key].item(), count  )


    def print_loss(self,count):
        print( str(count)+' iter finshed' )
        for key in self.loss_dict:
            print(key, self.loss_dict[key].item())
        print(' ')


    def visualize(self, count):
        self.encoder.eval()
        with torch.no_grad():   
            w_plus_latent = self.encoder( self.train_sample['real_seg'] ) + self.mean_latent
            fake_img, _ = self.generator(w_plus_latent, input_type='w+')              
            save_path = os.path.join( 'output', self.args.name, 'sample_train', str(count).zfill(6)+'.png' )
            utils.save_image( fake_img, save_path, nrow=int(self.args.n_sample**0.5), normalize=True )

            w_plus_latent = self.encoder( self.test_sample['real_seg'] ) + self.mean_latent
            fake_img, _ = self.generator(w_plus_latent, input_type='w+')    
            save_path = os.path.join( 'output', self.args.name, 'sample_test', str(count).zfill(6)+'.png' )
            utils.save_image( fake_img, save_path, nrow=int(self.args.n_sample**0.5), normalize=True )
        
        self.encoder.train()


    def save_ckpt(self, count):
        # TODO: now only save encoder, which is not compitible with self.load_ckpt code
        save_dict = {"e": self.encoder.state_dict(),"args": self.args}
        save_path = os.path.join( 'output', self.args.name, 'checkpoint', str(count).zfill(6)+'.pt' )
        torch.save(save_dict, save_path)
        old_ckpt = os.path.join( 'output', self.args.name, 'checkpoint', str(count-10000).zfill(6)+'.pt' )
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


    def cal_E_loss(self, fake_img, target_img, latent=None):
        loss = 0 
        if self.args.w_norm_lambda>0:
            w_norm_loss = self.get_w_norm_loss(latent, self.mean_latent)*self.args.w_norm_lambda
            loss += w_norm_loss
            self.loss_dict['w_norm_loss'] = w_norm_loss
        if self.args.l2_lambda>0:
            l2_loss = self.get_l2_loss( fake_img, target_img )*self.args.l2_lambda
            loss += l2_loss
            self.loss_dict['l2_loss'] = l2_loss
        if self.args.lpips_lambda>0:        
            lpips_loss = self.get_lpips_loss( fake_img, target_img )*self.args.lpips_lambda
            loss += lpips_loss
            self.loss_dict['lpips_loss'] = lpips_loss
        return loss



    def trainE(self):

        real_img, real_seg = process_data( next(self.train_loader), self.device )
    
        # get w+ code and forward G
        w_plus_latent = self.encoder(real_seg) # bs*14*512
        w_plus_latent += self.mean_latent
        fake_img, _ = self.generator(w_plus_latent, input_type='w+')
        
        loss = self.cal_E_loss( fake_img, real_img, w_plus_latent )
        self.optimizerE.zero_grad()
        loss.backward()
        self.optimizerE.step()
    
    
    def train(self): 

        for idx in range(self.args.iter):
            count = idx + self.args.start_iter

            if count > self.args.iter:
                print("Done!")
                break

            self.trainE()
            self.write_loss(count)
          
            if count % 50 == 0:
                self.print_loss(count)
            if count % 100 == 0:
                self.visualize(count)   
            if count % 10000 == 0:  
                self.save_ckpt(count)







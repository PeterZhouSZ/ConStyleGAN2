import math
import random
import torch
from torch import nn
from torch.nn import functional as F


from models.stylegan2.building_blocks import PixelNorm, EqualLinear, ConstantInput, StyledConv, ConvLayer, ResBlock, ToRGB 

from models.utils import EdgeDetector, FeaturePropagator



class Generator(nn.Module):
    def __init__(self, args, device, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, ):
        super().__init__()

        self.args = args
        self.device = device

        if not args.deterministic: # if deterministic, we do not learn mapping network F 
            layers = [PixelNorm()]
            for i in range(args.n_mlp):
                layers.append( EqualLinear( args.style_dim, args.style_dim, lr_mul=lr_mlp, activation='fused_lrelu' )  )
            self.style = nn.Sequential(*layers)


        self.channels = {   4: 512,
                            8: 512,
                            16: 512,
                            32: 512,
                            64: 256 * args.channel_multiplier,
                            128: 128 * args.channel_multiplier,
                            256: 64 * args.channel_multiplier,
                            512: 32 * args.channel_multiplier,
                            1024: 16 * args.channel_multiplier }

        self.encoder = Encoder(args)

        self.log_size = int(math.log(args.data_size, 2)) - int(math.log(self.args.starting_feature_size_instance, 2))
        self.num_layers = self.log_size * 2 
        self.n_latent = self.log_size * 2 + 1 

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        expected_out_size = self.args.starting_feature_size_instance
        layer_idx = 0 
        for _ in range(self.log_size):
            expected_out_size *= 2
            shape = [1, 1, expected_out_size, expected_out_size]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.zeros(*shape))
            self.noises.register_buffer(f'noise_{layer_idx+1}', torch.zeros(*shape))
            layer_idx += 2 

        in_channel = self.channels[self.args.starting_feature_size_instance]
        expected_out_size = self.args.starting_feature_size_instance      
        for _ in range(self.log_size):  
            expected_out_size *= 2 
            out_channel = self.channels[expected_out_size]
            self.convs.append( StyledConv( in_channel, out_channel, 3, args.style_dim, upsample=True, blur_kernel=blur_kernel ) )
            self.convs.append( StyledConv(out_channel, out_channel, 3, args.style_dim, blur_kernel=blur_kernel) )
            self.to_rgbs.append(ToRGB(out_channel, args.style_dim))
            in_channel = out_channel
        

        if self.args.propagate:
            self.detector = EdgeDetector()
            self.propagator_container =   { 4: FeaturePropagator(4, args.propagator_kernel).to(device),
                                            8: FeaturePropagator(8, args.propagator_kernel).to(device),
                                            16: FeaturePropagator(16, args.propagator_kernel).to(device),
                                            32: FeaturePropagator(32, args.propagator_kernel).to(device),
                                            64: FeaturePropagator(64, args.propagator_kernel).to(device),
                                            128: FeaturePropagator(128, args.propagator_kernel).to(device),
                                            256: FeaturePropagator(256, args.propagator_kernel).to(device),
                                            512: FeaturePropagator(512, args.propagator_kernel).to(device) }
                               
              

    def make_noise(self):
        expected_out_size = self.args.starting_feature_size_instance
        for _ in range(self.log_size):
            expected_out_size *= 2
            noises.append( torch.randn(1, 1, expected_out_size, expected_out_size, device=self.device) )
            noises.append( torch.randn(1, 1, expected_out_size, expected_out_size, device=self.device) )
        return noises


    def mean_latent(self, n_latent):
        latent_in = torch.randn( n_latent, self.args.style_dim, device=self.device)
        latent = self.style(latent_in).mean(0, keepdim=True).unsqueeze(1)
        return latent


    def get_latent(self, input):
        return self.style(input)


    def __prepare_starting_feature(self, global_pri, styles, input_type):
        feature, z, loss = self.encoder(global_pri)
        if input_type == None:
            styles = [z]
            input_type = 'w' if self.args.deterministic else 'z'
        return  feature, styles, input_type, loss

        
    def __prepare_letent(self, styles, inject_index, truncation, truncation_latent,  input_type):
        "This is a private function to prepare w+ space code needed during forward"

        if input_type == 'z':
            styles = [self.style(s).unsqueeze(1) for s in styles]  # each one is bs*1*512
        elif input_type == 'w':
            styles = [s.unsqueeze(1) for s in styles]  # each one is bs*1*512
        else: 
            return styles # w+ case 

        # truncate each w 
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append( truncation_latent + truncation * (style-truncation_latent)  )
            styles = style_t

        # duplicate and concat into BS * n_latent * code_len 
        if len(styles) == 1:
            latent = styles[0].repeat(1, self.n_latent, 1)  
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent1 = styles[0].repeat(1, inject_index, 1)
            latent2 = styles[1].repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        else:
            latent = torch.cat( styles, 1 )

        return latent


    def __prepare_noise(self, noise, randomize_noise):
        "This is a private function to prepare noise needed during forward"
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [ getattr(self.noises, f'noise_{i}') for i in range(self.num_layers) ]
        return noise


    def modify_feature(self, x, mask):
        if  not self.args.propagate or mask == None:
            return x 
        else:
            featurepropagator = self.propagator_container[ x.shape[1] ]
            mask =  F.interpolate(mask, size=x.size()[2:] )
            edge = self.detector( mask, 'tight'  )
            return  featurepropagator(x, mask-edge, edge)


    
    def forward(self, global_pri, target_seg=None, styles=None, return_latents=False, inject_index=None, truncation=1, truncation_latent=None, input_type=None, noise=None, randomize_noise=True, return_loss=True):

        """
        global_pri: a tensor with the shape BS*C*self.prior_size*self.prior_size

        styles: it should be either list or a tensor or None.
                List Case:
                    a list containing either z code or w code.
                    and each code (whether z or w, specify by input_type) in this list should be bs*len_of_code.
                    And number of codes should be 1 or 2 or self.n_latent. 
                    When len==1, later this code will be broadcast into bs*self.n_latent*512
                    if it is 2 then it will perform style mixing. If it is self.n_latent, then each of them will 
                    provide style for each layer.
                Tensor Case:
                    then it has to be bs*self.n_latent*code_len, which means it is a w+ code.
                    In this case input_type should be 'w+', and for now we do not support truncate,
                    we assume the input is a ready-to-go latent code from w+ space
                None Case:
                    Then z code will be derived from global_pri also. In this case input_type shuold be None

        target_seg: a binary tensor (BS*1*H*W) specifying fg object shape. It will be used in modify_feature. 
                    If None then args.modify_feature should also be 'none', which means feature will not be modified 
                    according to shape mask.  
            
        return_latents: if true w+ code: bs*self.n_latent*512 tensor, will be returned 

        inject_index: int value, it will be specify for style mixing, only will be used when len(styles)==2 

        truncation: whether each w will be truncated 
        
        truncation_latent: if given then it should be calculated from mean_latent function. It has size 1*1*512
                           if truncation, then this latent must be given 

        input_type: input type of styles, None, 'z', 'w' 'w+'
        
        noise: if given then recommand to run make_noise first to get noise and then use that as input. if given 
               randomize_noise will be ignored 
         
        randomize_noise: if true then each forward will use different noise, if not a pre-registered fixed noise
                         will be used for each forward.

        return_loss: if return kl loss.  

        """
        if input_type == 'z' or input_type == 'w': 
            assert len(styles) in [1,2,self.n_latent], 'number of styles must be 1, 2 or self.n_latent'
        elif input_type == 'w+':
            assert styles.ndim == 3 and styles.shape[1] == self.n_latent
        elif input_type == None:
            assert styles == None
        else:
            assert False, 'not supported input_type'

        
        start_feature, styles, input_type, loss = self.__prepare_starting_feature(global_pri, styles, input_type)
        latent = self.__prepare_letent(styles, inject_index, truncation, truncation_latent, input_type)
        noise = self.__prepare_noise(noise, randomize_noise)
    
        # # # start generating # # #  

        out = start_feature
        skip = None

        i = 0
        for conv1, conv2, noise1, noise2, to_rgb in zip( self.convs[::2], self.convs[1::2], noise[::2], noise[1::2], self.to_rgbs ):
            out = self.modify_feature(out, target_seg)
            out = conv1(out, latent[:, i], noise=noise1)
            out = self.modify_feature(out, target_seg)
            out = conv2(out, latent[:, i + 1], noise=noise2)
   
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
            
        image = F.tanh(skip)

        output = { 'image': image }
        if return_latents:
            output['latent'] =  latent  
        if return_loss:
            output['klloss'] =  loss 

        return output







class Discriminator(nn.Module):
    def __init__(self, size, args, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * args.channel_multiplier,
            128: 128 * args.channel_multiplier,
            256: 64 * args.channel_multiplier,
            512: 32 * args.channel_multiplier,
            1024: 16 * args.channel_multiplier,
        }

        log_size = int( math.log(size,2) )
        convs = [ ConvLayer(3, channels[size], 1) ]        

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential( EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
                                           EqualLinear(channels[4], 1) )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view( group, -1, self.stddev_feat, channel // self.stddev_feat, height, width )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out










class SmallUnet(nn.Module):
    "Here we aggregate feature from different resolution. It is actually the up branch of Unet"
    def __init__(self, size_to_channel):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        size = 4 
        extra_channel = 0 
        
        for i in range(  len(size_to_channel)  ):
            in_channel = size_to_channel[size] + extra_channel
            upsample = i != (len(size_to_channel)-1) 
            self.convs.append(     ConvLayer(in_channel, 512, 3, upsample=upsample)      )
            size *= 2
            extra_channel = 512
    
    def forward(self, feature_list):
        "feature_list should be ordered from small to big: BS*C1*4*4, BS*C2*8*8, BS*C3*16*16,..."
     
        for conv, feature in zip(self.convs, feature_list):
            if feature.shape[3] != 4:  
                feature = torch.cat( [feature,previos], dim=1 )
            previos = conv(feature)
            
        return previos



# class Encoder(nn.Module):
#     def __init__(self, args, blur_kernel=[1, 3, 3, 1]):
#         super().__init__()
#         self.args = args

#         channels = { 4: 512,
#                      8: 512,
#                      16: 512,
#                      32: 512,
#                      64: 512,
#                      128: 128 * args.channel_multiplier,
#                      256: 64 * args.channel_multiplier,
#                      512: 32 * args.channel_multiplier,
#                      1024: 16 * args.channel_multiplier }

#         in_c = args.number_of_semantic if args.have_zero_class else args.number_of_semantic + 1 # this 1 is extra padded 0 class 
#         self.convs1 = ConvLayer(3+in_c+1, channels[args.prior_size], 1) # this 1 is instance segmentation mask 

#         log_size = int(math.log(args.prior_size, 2))

#         in_channel = channels[args.prior_size]
#         size_to_channel = {} # indicate from which resolution we provide spatial feature 
#         self.convs2 = nn.ModuleList()
#         for i in range(log_size, 2, -1):
#             out_size = 2 ** (i-1)
#             out_channel = channels[out_size]
#             if 4 <= out_size//2 <= args.starting_feature_size_instance: # cropped center is needed feature, thus why //2
#                 size_to_channel[out_size//2] = out_channel 
#             self.convs2.append(ResBlock(in_channel, out_channel, blur_kernel))
#             in_channel = out_channel

#         self.unet = SmallUnet( size_to_channel )


#         self.final_linear = EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu')
#         self.mu_linear = EqualLinear(channels[4], args.style_dim)
#         self.var_linear = EqualLinear(channels[4], args.style_dim)


#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         flag = 0 if self.args.deterministic else 1 # if deterministic variance branch will not be updated
#         eps = torch.randn_like(std)*flag    
#         return eps.mul(std) + mu
    

#     def get_kl_loss(self, mu, logvar):
#         return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


#     def crop_feature(self, x):
#         size = x.shape[3]
#         start = size // 4 
#         end = start + size // 2 
#         return x[ :, :, start:end, start:end ]


#     def forward(self, input):
#         batch = input.shape[0]
#         intermediate_feature = []

#         out = self.convs1(input)
#         for conv in self.convs2:
#             out = conv(out)
#             cropped_feature_size = out.shape[3] // 2
#             if 4 <= cropped_feature_size <= self.args.starting_feature_size_instance:
#                 intermediate_feature.append( self.crop_feature(out) )  
#         feature = self.unet( intermediate_feature[::-1] )


#         out = self.final_linear( out.view(batch, -1)  )
#         mu = self.mu_linear(out)
#         logvar = self.var_linear(out)
#         z = self.reparameterize(mu, logvar)

#         flag = 0 if self.args.deterministic else 1
#         loss = self.get_kl_loss(mu, logvar)*flag

#         return feature, z, loss








# class Encoder(nn.Module):
#     def __init__(self, args, blur_kernel=[1, 3, 3, 1]):
#         super().__init__()
#         self.args = args
#         print('THIS IS NON CONTEXT CROPPING ENCODER')
#         channels = { 4: 512,
#                      8: 512,
#                      16: 512,
#                      32: 512,
#                      64: 512,
#                      128: 128 * args.channel_multiplier,
#                      256: 64 * args.channel_multiplier,
#                      512: 32 * args.channel_multiplier,
#                      1024: 16 * args.channel_multiplier }

#         in_c = args.number_of_semantic if args.have_zero_class else args.number_of_semantic + 1 # this 1 is extra padded 0 class 
#         self.convs1 = ConvLayer(3+in_c+1, channels[args.prior_size], 1) # this 1 is instance segmentation mask 

#         log_size = int(math.log(args.prior_size, 2))

#         in_channel = channels[args.prior_size]
#         size_to_channel = {} # indicate from which resolution we provide spatial feature 
#         self.convs2 = nn.ModuleList()
#         for i in range(log_size, 2, -1):
#             out_size = 2 ** (i-1)
#             out_channel = channels[out_size]
#             if 4 <= out_size <= args.starting_feature_size_instance: # cropped center is needed feature, thus why //2
#                 size_to_channel[out_size] = out_channel 
#             self.convs2.append(ResBlock(in_channel, out_channel, blur_kernel))
#             in_channel = out_channel

#         self.unet = SmallUnet( size_to_channel )


#         self.final_linear = EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu')
#         self.mu_linear = EqualLinear(channels[4], args.style_dim)
#         self.var_linear = EqualLinear(channels[4], args.style_dim)


#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         flag = 0 if self.args.deterministic else 1 # if deterministic variance branch will not be updated
#         eps = torch.randn_like(std)*flag    
#         return eps.mul(std) + mu
    

#     def get_kl_loss(self, mu, logvar):
#         return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


#     def crop_feature(self, x):
#         return x


#     def forward(self, input):
#         batch = input.shape[0]
#         intermediate_feature = []

#         out = self.convs1(input)
#         for conv in self.convs2:
#             out = conv(out)
#             cropped_feature_size = out.shape[3]
#             if 4 <= cropped_feature_size <= self.args.starting_feature_size_instance:
#                 intermediate_feature.append( self.crop_feature(out) )  
#         feature = self.unet( intermediate_feature[::-1] )


#         out = self.final_linear( out.view(batch, -1)  )
#         mu = self.mu_linear(out)
#         logvar = self.var_linear(out)
#         z = self.reparameterize(mu, logvar)

#         flag = 0 if self.args.deterministic else 1
#         loss = self.get_kl_loss(mu, logvar)*flag

#         return feature, z, loss






"This is the combined version of 2x encoder and no context encoder"
class Encoder(nn.Module):
    def __init__(self, args, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.args = args

        channels = { 4: 512,
                     8: 512,
                     16: 512,
                     32: 512,
                     64: 512,
                     128: 128 * args.channel_multiplier,
                     256: 64 * args.channel_multiplier,
                     512: 32 * args.channel_multiplier,
                     1024: 16 * args.channel_multiplier }

        in_c = args.number_of_semantic if args.have_zero_class else args.number_of_semantic + 1 # this 1 is extra padded 0 class 
        self.convs1 = ConvLayer(3+in_c+1, channels[args.prior_size], 1) # this 1 is instance segmentation mask 

        log_size = int(math.log(args.prior_size, 2))

        # this is used to differentiate if no context or 2 times context
        self.ratio = args.prior_size // args.center_size

        in_channel = channels[args.prior_size]
        size_to_channel = {} # indicate from which resolution we provide spatial feature 
        self.convs2 = nn.ModuleList()
        for i in range(log_size, 2, -1):
            out_size = 2 ** (i-1)
            out_channel = channels[out_size]
            if 4 <= out_size//self.ratio <= args.starting_feature_size_instance: # cropped center is needed feature, thus why //2
                size_to_channel[out_size//self.ratio] = out_channel 
            self.convs2.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.unet = SmallUnet( size_to_channel )


        self.final_linear = EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu')
        self.mu_linear = EqualLinear(channels[4], args.style_dim)
        self.var_linear = EqualLinear(channels[4], args.style_dim)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        flag = 0 if self.args.deterministic else 1 # if deterministic variance branch will not be updated
        eps = torch.randn_like(std)*flag    
        return eps.mul(std) + mu
    

    def get_kl_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    def crop_feature(self, x):
        if self.ratio == 2:
            # in this case crop the center feature
            size = x.shape[3]
            start = size // 4 
            end = start + size // 2 
            return x[ :, :, start:end, start:end ]
        else:
            return x 


    def forward(self, input):
        batch = input.shape[0]
        intermediate_feature = []

        out = self.convs1(input)
        for conv in self.convs2:
            out = conv(out)
            cropped_feature_size = out.shape[3]//self.ratio
            if 4 <= cropped_feature_size <= self.args.starting_feature_size_instance:
                intermediate_feature.append( self.crop_feature(out) )  
        feature = self.unet( intermediate_feature[::-1] )
        

        out = self.final_linear( out.view(batch, -1)  )
        mu = self.mu_linear(out)
        logvar = self.var_linear(out)
        z = self.reparameterize(mu, logvar)

        flag = 0 if self.args.deterministic else 1
        loss = self.get_kl_loss(mu, logvar)*flag

        return feature, z, loss



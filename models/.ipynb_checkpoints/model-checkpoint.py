import math
import random
import torch
from torch import nn
from torch.nn import functional as F


from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        #  first **2, then avg among C channel, then 1/sqrt() 
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__( self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False, blur_kernel=[1,3,3,1] ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter( torch.randn(1, out_channel, in_channel, kernel_size, kernel_size) )
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def __repr__(self):
        "For print"
        return ( f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})' )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        
        # apply afine transformation on style 
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view( batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view( batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size )
            weight = weight.transpose(1, 2).reshape( batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):

        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        # input is only used for get batch size 
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True ):
        super().__init__()

        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate)
        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out








class Generator(nn.Module):
    def __init__(self, args, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01 ):
        super().__init__()


        self.encoder = Encoder(args)

        # mapping function of styleGAN
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

        #self.input = ConstantInput(self.channels[4]) # a learnable constance 1*512*4*4 

        self.conv1 = StyledConv( self.channels[4], self.channels[4], 3, args.style_dim, blur_kernel=blur_kernel )
        self.to_rgb1 = ToRGB(self.channels[4], args.style_dim, upsample=False)

        self.log_size = int(math.log(args.data_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.n_latent = self.log_size * 2 - 2

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.avg_noises = nn.Module()
        

        # avg noise
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.avg_noises.register_buffer(f'noise_{layer_idx}', torch.zeros(*shape))


        # generatoe mdule
        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append( StyledConv( in_channel, out_channel, 3, args.style_dim, upsample=True, blur_kernel=blur_kernel ) )
            self.convs.append( StyledConv(out_channel, out_channel, 3, args.style_dim, blur_kernel=blur_kernel) )
            self.to_rgbs.append(ToRGB(out_channel, args.style_dim))
            in_channel = out_channel


    def sample_norm_from_this_prior(self, input):
        """Input is rgb and semantic prior, 
           we sample from encoded dist and return z 
        """
        _, z, _ = self.encoder(input, return_loss=False)
        return z 



    def forward(self, input, mask, return_loss, input_is_norm=False, noise=None, randomize_noise=False):
        """ 
        During training the input is supposed to be the image and semantic prior,
        but since we force to follow norm dist, we may be able to sample from it 
        """

        # first get w vector (call it latent here)
        if input_is_norm:
            assert False, "current implementataion is for non-constant"
            assert return_loss == False, "input is sampled from norm dist, kl loss can not be calculated"
            styles = self.style(input) 
        else:
            feature, z, kl_loss = self.encoder(input, return_loss)
            styles = self.style(z) 
        latent = styles.unsqueeze(1).repeat(1, self.n_latent, 1)


        # then get noise 
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [ getattr(self.avg_noises, f'noise_{i}') for i in range(self.num_layers) ]


        # This is beginning of generator. what we have here: nonconstant, latent(bs*n_latent*w_len), noise

        out = feature # we use non-constant feature from E. it was:  self.input(latent) 
        out = self.conv1(out, latent[:,0], noise=0) # we do not inject noise on this encoder's feature        
        skip = self.to_rgb1(out, latent[:, 1])


        # a = [a,b,c,d,e,f],  a[::2]=[a,c,e],  a[1::2]=[b,d,f]    a[2::2]=[c,e]  you get it 
        i = 1                                                        
        for conv1, conv2, noise1, noise2, to_rgb in zip( self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs ):
            
            out = out * F.interpolate(mask, size=out.size()[2:], mode='bilinear')
            out = conv1(out, latent[:, i], noise=noise1)

            out = out * F.interpolate(mask, size=out.size()[2:], mode='bilinear')
            out = conv2(out, latent[:, i + 1], noise=noise2)
        
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = F.tanh(skip)
        
        if return_loss:
            return image, kl_loss
        else:
            return image



#################################################################################



class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, args, size, blur_kernel=[1, 3, 3, 1]):
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

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential( *convs[:-1] )
        self.convs_rf_head = nn.Conv2d(512,1,1) 

        self.convs_extra = convs[-1]
        

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input, return_patch_loss=False):
        "If not return patch loss this is same as original D"

        out = self.convs(input)  # feature size 8*8 
        patch_rf = self.convs_rf_head(out)

        out = self.convs_extra(out) # feature size 4*4 
        

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        

        out = out.view(batch, -1)
        out = self.final_linear(out)

        if return_patch_loss:
            return patch_rf, out
        else:
            return out 







class Encoder(nn.Module):
    def __init__(self, args, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = { 4: 512,
                     8: 512,
                     16: 512,
                     32: 512,
                     64: 256 * args.channel_multiplier,
                     128: 128 * args.channel_multiplier,
                     256: 64 * args.channel_multiplier,
                     512: 32 * args.channel_multiplier,
                     1024: 16 * args.channel_multiplier }

        convs = [ConvLayer(3+151+1, channels[args.prior_size], 1)]

        log_size = int(math.log(args.prior_size, 2))

        in_channel = channels[args.prior_size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)


        self.final_linear = EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu')
        self.mu_linear = EqualLinear(channels[4], args.style_dim)
        self.var_linear = EqualLinear(channels[4], args.style_dim)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return eps.mul(std) + mu
    
    def get_kl_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    def forward(self, input, return_loss):
        batch = input.shape[0]
        feature = self.convs(input)
        out = feature.view(batch, -1)
        out = self.final_linear(out)

        mu = self.mu_linear(out)
        logvar = self.var_linear(out)
        z = self.reparameterize(mu, logvar)

        if not return_loss:
            return feature, z, None
        else:
            loss = self.get_kl_loss(mu, logvar)
            return feature, z, loss
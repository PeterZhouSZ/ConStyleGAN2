import torch.nn.functional as F
import torch 
import torchvision 
import torch.nn as nn


class Merger():
    def __init__(self, args, device, ignore=True):
        """For the meaning of ignore see EXPLANTION_5 in dataset/instance_dataset/process_funcs.py"""

        self.args = args
        self.device = device
        self.ignore = ignore

        self.rgb_enlarger = RGB_Enlarger().to(device)
        self.mask_enlarger = Mask_Enlarger().to(device)
        


        if ignore:
            self.new_size = [args.center_size, args.center_size]
            start = (args.prior_size - args.center_size) / 2 
            end = start + args.center_size
            self.new_box = [ int(start), int(start), int(end), int(end) ]



    def relocate_one_shot(self, fake_img, real_seg, info, global_shape):
        "Here we ingore small offset so info will be ingored, instead we use self.new_box and self.new_size"

        bs = fake_img.shape[0]
     
        # resize this img and seg. default is nearest, so seg still be binary
        fake_img = F.interpolate( fake_img, size=self.new_size )
        real_seg = F.interpolate( real_seg, size=self.new_size )    
        # create a temp canvas  
        temp_img = torch.zeros( bs, 3, global_shape[2], global_shape[3] ).to(self.device) 
        temp_seg = torch.zeros( bs, 1, global_shape[2], global_shape[3] ).to(self.device)
        # relocate   
        x1,y1,x2,y2 = self.new_box
        temp_img[:, :, y1:y2, x1:x2 ] = fake_img
        temp_seg[:, :, y1:y2, x1:x2 ] = real_seg
        
        return temp_img, temp_seg
        



    def relocate_one_by_one(self, fake_imgs, real_segs, infos, global_shape):

        new_fake_img, new_real_seg = [],[]    

        for i in range( fake_imgs.shape[0] ):

            fake_img, real_seg, info = fake_imgs[i], real_segs[i], infos[i]   
            x1,y1,x2,y2 = info['new_box']
            # resize this img and seg (default is nearest, so seg still be binary)
            fake_img = F.interpolate(fake_img.unsqueeze(0), size=info['new_size']).squeeze(0)
            real_seg = F.interpolate(real_seg.unsqueeze(0), size=info['new_size']).squeeze(0)       
            # create a temp canvas  
            temp_img = torch.zeros( 3, global_shape[2], global_shape[3] ).to(self.device)
            temp_seg = torch.zeros( 1, global_shape[2], global_shape[3] ).to(self.device)
            # relocate  
            temp_img[:, y1:y2, x1:x2 ] = fake_img
            temp_seg[:, y1:y2, x1:x2 ] = real_seg

            new_fake_img.append(temp_img)
            new_real_seg.append(temp_seg)
       
        return torch.stack(new_fake_img), torch.stack(new_real_seg)




    def __call__(self, fake_img, real_seg, global_img, global_seg, info):
        """
        fake_img is output from our fg generator
        real_seg is the one used to generate fake_img and it will be used in enlarger to expand pixels of fake_img
        global_img is base image where fake img will be put on
        global_seg is mask for composition  
        info stores size and box infomation where to put fg 

        Note that: although we have real_seg and we know the size and location of fg in global_img, so we can derive
        a global_seg also (call it global_seg2 later). But global_seg2 is actually not exactly same as global_seg due to
        multiple resize and quantization. Whereas global_seg is actually 100% accurately aligned with fg obj in  global_img
        so we will use it to do 'put it back' operation.
        """

        # relocate 
        if self.ignore:
            fake_img, real_seg = self.relocate_one_shot(fake_img, real_seg, info, global_img.shape)
        else:
            fake_img, real_seg = self.relocate_one_by_one(fake_img, real_seg, info, global_img.shape)
            
         
        # dilate fake_img and global_seg 
        dialted_fake_img = self.rgb_enlarger(fake_img, real_seg)
        alpha_mask = self.mask_enlarger(global_seg, 0, 3) 
        

        # composition         
        output = dialted_fake_img*alpha_mask + global_img*(1-alpha_mask)

        return output
        










class Mask_Enlarger(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,1,3,1,1, bias=False)
        self.conv.weight.data.fill_(1/9)
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, input, hard=1, soft=1):
        """
        soft means we apply 3*3 conv, and boundary values will have non-binaray value
        hard means after 3*3 conv as long as value is non zero, we will convert it into 1 
        """

        if hard>0:
            x = input
            for _ in range(hard):
                x = self.conv(x) 
                x[x!=0] = 1 

        if soft>0:
            x = x if hard>0 else input
            for _ in range(soft):
                x = self.conv(x)
        
        if hard + soft > 1:     
            return x #torch.clamp(input+x ,0 ,1)
        else:
            return input




class PartialConv(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        "Bigger kernel size are, more pixels will be dialated"
        
        assert kernel_size in [3,5,7]
        padding = kernel_size // 2   
        
        self.kernel_size = kernel_size               
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, 1, padding, bias=False)        
        self.mask_conv.weight.data.fill_(1.0)
        
        self.input_conv = nn.Conv2d(3, 3, kernel_size, 1, padding, bias=False) 
        for i in range(3):
            init = torch.zeros(3,kernel_size,kernel_size)
            init[i,:,:] = 1/(kernel_size*kernel_size) 
            self.input_conv.weight.data[i] = init
                   
        for param in self.parameters():
            param.requires_grad = False
             
 
    def forward(self, input, mask, return_new_mask = False):

        output = self.input_conv( input*mask )
        mask_output = self.mask_conv(mask)

        no_update_holes = mask_output == 0
        mask_ratio = (self.kernel_size*self.kernel_size) / mask_output.masked_fill_(no_update_holes, 1.0)

        output = output * mask_ratio 
        output = output.masked_fill_(no_update_holes, 0.0)
        
        # restore the original infomation within the input mask 
        output = input*mask + output*(1-mask)
        
        if not return_new_mask:
            return output
        else:
            new_mask = 1-no_update_holes*1
            return output, new_mask.float()




class RGB_Enlarger(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Here I actually are trying to dialate image's boundary, buy I do not find other good ways to achieve this,
        so I implement this by using partial conv layer (fixed weight). See PartialConv for details.

        Note that bigger kernels you have in PartialConv, then each time you can dialate more pixels. 
        Roughly speeking if K = 3/5/7..., then you can increase thickness by 1/2/3 pixels (is it right?)
        But I do not recommend to use any number except than 3. Because larger kenel will also consider more 
        inner information when increase the boundary. For example in 1D, if you have:
        [0,  0,   0,      red,   yellow,   yellow], 
        and you use k=3, it should end up with 
        [0,  0,   red,    red,   yellow,   yellow ], 
        since you kenel can only see 'red'. 
        But if you have a kenel=5, it will have a result like this:
        [0,  0,   orange, red,   yellow,   yellow ],
        this is because your kernal can see both red and yellow. 

        This is bad, since the purpose of enlarging our generated foreground instance is that: after resizing and 
        relocate, its shape will not necessary be matched with mask directly from the scene. (we call global_seg in Merger)
        So if global_seg is actually like this:
        [0   ,0    1,     1,      1,         1 ]
        then masked out region will be 
        [orange, red,   yellow,   yellow], when the color the vibrant, you can see a clear edge in 'red' color region.

        Thus why we use k=3 in PartialConv and do twice enlarge in __call__, rather than using k=5 and do once

        
        """
        self.enlarger = PartialConv(3)
 
    def __call__(self, x, mask):   
  
        enlarged_x, enlarged_mask = x, mask
        for _ in range(2): 
            enlarged_x, enlarged_mask = self.enlarger(enlarged_x, enlarged_mask, True)
        return enlarged_x 
        
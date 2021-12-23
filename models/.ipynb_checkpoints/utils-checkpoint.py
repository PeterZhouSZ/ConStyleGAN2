import torch 
import torch.nn as nn


class EdgeDetector():
    """
    It support three different type of edge: tight, inner, outer.
    
    Example:
          [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    Tight means output edge tightly touches you mask and it is still INSIDE of mask:
          [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          
    Outer means edge touches contour of your mask but it is OUTSIDE of it:
          [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
          [1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
          [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          
    Inner means (your mask - tight edge) and then perform tight edge detection again:
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    
    
    """
    
    
    def get_edge(self, x, edge_type):   
        "This is core of this class. edge result right before if statement is tight+outer edge"
        edge = torch.zeros_like(x)
        edge[:,:,:,1:] = edge[:,:,:,1:] | (x[:,:,:,1:] != x[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (x[:,:,:,1:] != x[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (x[:,:,1:,:] != x[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (x[:,:,1:,:] != x[:,:,:-1,:])
        if edge_type == 'tight':
            return edge & x
        if edge_type == 'outer':
            return edge & (~x)     

            
    
    def __call__(self, x, edge_type='tight'):
        """
        x should be a binary mask tensor with shape: N*1*H*W
        """
        x = x.int()
        
        if edge_type == 'tight':
            return self.get_edge(x, edge_type).float()
        elif edge_type == 'outer':
            return self.get_edge(x, edge_type).float()
        elif edge_type == 'inner':
            tight_edge = self.get_edge(x, 'tight')
            x = x - tight_edge 
            return self.get_edge(x, 'tight').float()





class FeatureInterpolator(nn.Module):
    def __init__(self, channel):
        super().__init__()
        
        self.mask_conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)        
        self.mask_conv.weight.data.fill_(1.0)
        
        self.conv = nn.Conv2d(channel, channel, 3, 1, 1, bias=False) 
        for i in range(channel):
            init = torch.zeros(channel,3,3)
            init[i,:,:] = 1
            self.conv.weight.data[i] = init
            
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, mask):
        """
        x: feature tensor with shape N*C*H*W
        
        mask: binary mask (should be edge) with shape N*1*H*W where 1 indicates region needs to be filled 
              in with new interpolated feature
        """
        
        mask = 1 - mask # 1 is valid region and 0 is the region needs to be filled in with new feautre 

        new_x = self.conv(x*mask)
        valid_count = self.mask_conv(mask)
        new_x = new_x / valid_count
        
        # recover valid region 
        x = x*mask + new_x*(1-mask)
        
        return x 





class FeaturePropagator(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super().__init__()
        "Bigger kernel size is, more pixels will be considered"
        
        assert kernel_size in [3,5,7]
        padding = kernel_size // 2   
        
        self.kernel_size = kernel_size               
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, 1, padding, bias=False)        
        self.mask_conv.weight.data.fill_(1.0)
        
        self.input_conv = nn.Conv2d(channel, channel, kernel_size, 1, padding, bias=False) 
        for i in range(channel):
            init = torch.zeros(channel,kernel_size,kernel_size)
            init[i,:,:] = 1/(kernel_size*kernel_size) 
            self.input_conv.weight.data[i] = init
                   
        for param in self.parameters():
            param.requires_grad = False
            
 
    def forward(self, x, valid_mask, affect_mask):
        """
        x: input feature with the size of N*C*H*W
        
        valid_mask: a binary mask indicating which region has valid pixels/features N*1*H*W
        
        affect_mask: a binary mask indicating which region should be affcted by new feature, 
                     other region will be restored into orginal value
                    
        """

        output = self.input_conv( x*valid_mask )
        mask = self.mask_conv(valid_mask)

        no_update_holes = mask == 0
        mask_ratio = (self.kernel_size*self.kernel_size) / mask.masked_fill_(no_update_holes, 1.0)
        output = output * mask_ratio 
 
        return output*affect_mask + x*(1-affect_mask)
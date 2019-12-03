import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision import models
# from model.network_mm import Bottleneck


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):

        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        
        out = x + self.conv_block(x)  # add skip connections
        return out


"""TreeResnetGenerator"""
class DilatedResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):

        super(DilatedResnetBlock, self).__init__()
        self.conv_block1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout=use_dropout, use_bias=use_bias, num_dilation=1, num_padding=1)
        self.conv_block2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout=use_dropout, use_bias=use_bias, num_dilation=2, num_padding=2)
        self.conv_block3 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout=use_dropout, use_bias=use_bias, num_dilation=3, num_padding=3)
        self.conv_block4 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout=use_dropout, use_bias=use_bias, num_dilation=4, num_padding=4)
        
        self.joint1 = nn.Sequential(
                        nn.Conv2d(dim*2, dim*2, kernel_size=1, padding=0, bias=use_bias)
        )

        self.joint2 = nn.Sequential(
                nn.Conv2d(dim*4, dim, kernel_size=1, padding=0, bias=use_bias)
        )
    def build_conv_block(self, dim, padding_type, norm_layer, num_dilation, num_padding, use_dropout, use_bias):
        
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(num_padding)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(num_padding)]
        elif padding_type == 'zero':
            p = num_padding
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=num_dilation), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(num_padding)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(num_padding)]
        elif padding_type == 'zero':
            p = num_padding
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=num_dilation), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        child1 = self.conv_block1(x)
        child2 = self.conv_block2(x)
        child3 = self.conv_block3(x)
        child4 = self.conv_block4(x)

        node1 = self.joint1(torch.cat((child1, child2), dim=1))
        node2 = self.joint1(torch.cat((child3, child4), dim=1))
        node = self.joint2(torch.cat((node1, node2), dim=1))
        out = x + node  # add skip connections
        return out

class TreeResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', with_tanh=True):

        assert(n_blocks >= 0)
        super(TreeResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [DilatedResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if with_tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class UnetBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False, with_tanh=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if with_tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""
    
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, with_tanh=True):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """

        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, with_tanh=with_tanh)  # add the innermost layer
        # unet_block = InnerBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, with_tanh=with_tanh)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, with_tanh=with_tanh)
        unet_block = UnetBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, with_tanh=with_tanh)
        unet_block = UnetBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, with_tanh=with_tanh)
        self.model = UnetBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, with_tanh=with_tanh)  # add the outermost layer
    def forward(self, input):
        """Standard forward"""
        return self.model(input)










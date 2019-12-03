import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torchvision import models
import numpy as np
from .transforms import create_part

# class AttentionLoss(nn.Module):
#     def __init__(self):
#         super(AttentionLoss, self).__init__()

#     def forward(self):
        
#         att_loss = 
#         return att_loss

class NewL1Loss(nn.Module):
    def __init__(self):
        super(NewL1Loss, self).__init__()

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        max_diff = torch.max(diff)
        weight = diff / max_diff
        loss = weight * diff
        return loss.mean()

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        '''
        If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers.
        Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        buffer will be saved while loading buffer ===
        class Test(nn.Module):
            def __init__(self, module):
                super(Test, self).__init__()
                self.module = module
                self.register_param()
            
            def register_param():
                exist_w = hasattr(self.module, 'w')
                if not exist_w:
                    w = nn.Parameter(torch.ones(1))
                    self.module.register_parameter(w) # register 'w' to module

            def forward(self, x)
                return x
        '''
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    
    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            if isinstance(prediction[0], list):
                loss = 0
                for pred in prediction:
                    pred_ = pred[-1]
                    target_tensor = self.get_target_tensor(pred_, target_is_real)
                    loss += self.loss(pred_, target_tensor)
                return loss
            else:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
                
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()

        return loss

class PixelWiseBCELoss(nn.Module):
    def __init__(self, weight):
        super(PixelWiseBCELoss, self).__init__()
        self.weight = weight
        self.loss = BCEWithLogitsLoss()
    def forward(self, pred, target):
        loss = 0
        # per channel
        for index in range(pred.size(1)):
            loss += self.weight[index] * self.loss(pred[:,index,:,:], target[:,index,:,:])
        return loss

class PixelSoftmaxLoss(nn.Module):
    def __init__(self, weight):
        super(PixelSoftmaxLoss, self).__init__()
        self.loss = CrossEntropyLoss(weight=weight)
    def forward(self, pred, target):
        pred = pred.reshape(pred.size(0), pred.size(1), -1) # batch, num_class, size, size
        _ , pos = torch.topk(target, 1, 1, True)
        pos = pos.reshape(pos.size(0), -1)
        loss = self.loss(pred, pos)
        return loss
        
class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)


class NNLoss(nn.Module):

    def __init__(self):
        super(NNLoss, self).__init__()
        
    def forward(self, predicted, ground_truth, nh=5, nw=5):
        v_pad = nh // 2
        h_pad = nw // 2
        val_pad = nn.ConstantPad2d((v_pad, v_pad, h_pad, h_pad), -10000)(ground_truth)

        reference_tensors = []
        for i_begin in range(0, nh):
            i_end = i_begin - nh + 1
            i_end = None if i_end == 0 else i_end
            for j_begin in range(0, nw):
                j_end = j_begin - nw + 1
                j_end = None if j_end == 0 else j_end
                sub_tensor = val_pad[:, :, i_begin:i_end, j_begin:j_end]
                reference_tensors.append(sub_tensor.unsqueeze(-1))
        
        reference = torch.cat(reference_tensors, dim=-1)
        ground_truth = ground_truth.unsqueeze(dim=-1)

        predicted = predicted.unsqueeze(-1)
        # return reference, predicted
        abs = torch.abs(reference - predicted)
        # sum along channels
        norms = torch.sum(abs, dim=1)
        # min over neighbourhood
        loss,_ = torch.min(norms, dim=-1)
        # loss = torch.sum(loss)/self.batch_size
        loss = torch.mean(loss)
        return loss
# vgg19 
# TO DO: whether should we add normalization
class VGGLoss(torch.nn.Module):
    def __init__(self, model_path, requires_grad=False):
        super(VGGLoss, self).__init__()
        self.model = models.vgg19()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model = torch.nn.DataParallel(self.model).cuda()
        vgg_pretrained_features = self.model.module.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.loss = nn.L1Loss().cuda()
        self.lossmse = nn.MSELoss().cuda()
        self.norm = FeatureL2Norm().cuda()  # before cal loss
        self.nnloss = NNLoss().cuda()
        # vgg19_bn: 53 layers || vgg19 : 37 layers || vgg19 2-7-12-21-30  before relu
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x]) # conv1_2
        for x in range(7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x]) # conv2_2
        for x in range(12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x]) # conv3_2
        for x in range(21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x]) # conv4_2
        for x in range(30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x]) # conv5_2
            
        self.slice = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]

        for i in range(len(self.slice)):
            self.slice[i] = torch.nn.DataParallel(self.slice[i]).cuda()
        # self.nnloss = torch.nn.DataParallel(self.loss)

        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False
        
    def forward(self, pred, target, target_parse, masksampled, gram, nearest, use_l1=True):

        weight = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] # more high level info
        weight.reverse()
        loss = 0
        # print(self.slice[0](pred).shape)
        if gram:
            loss_conv12 = self.lossmse(self.gram(self.slice[0](pred)), self.gram(self.slice[0](target)))
        elif nearest:
            loss_conv12 = self.nnloss(self.slice[0](pred), self.slice[0](target))
        else:
            loss_conv12 = self.loss(self.slice[0](pred), self.slice[0](target))
            # reference, predicted = self.loss(self.norm(self.slice[0](pred)), self.norm(self.slice[0](target)))
            # abs = torch.abs(reference - predicted)
            # # sum along channels
            # norms = torch.sum(abs, dim=1)
            # # min over neighbourhood
            # loss,_ = torch.min(norms, dim=-1)
            # # loss = torch.sum(loss)/self.batch_size
            # loss_conv12 = torch.mean(loss)
        
        for i in range(5):
            if not masksampled:
                if gram:
                    gram_pred = self.gram(self.slice[i](pred))
                    gram_target = self.gram(self.slice[i](target))
                else:
                    gram_pred = self.slice[i](pred)
                    gram_target = self.slice[i](target)
                if use_l1:
                    loss = loss + weight[i] * self.loss(gram_pred, gram_target)
                else:
                    loss = loss + weight[i] * self.lossmse(gram_pred, gram_target)
            else:
                pred = create_part(pred, target_parse, 'cloth')
                target = create_part(pred, target_parse, 'cloth')
                if gram:
                    gram_pred = self.gram(self.slice[i](pred))
                    gram_target = self.gram(self.slice[i](target))
                else:
                    gram_pred = self.slice[i](pred)
                    gram_target = self.slice[i](target)
                if use_l1:
                    loss = loss + weight[i] * self.loss(gram_pred, gram_target)
                else:
                    loss = loss + weight[i] * self.lossmse(gram_pred, gram_target)
        return loss, loss_conv12
    
    # Calculate Gram matrix (G = FF^T)
    def gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight * 2 * (h_tv/count_h + w_tv/count_w) / batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

if __name__ == "__main__":

    def gram(x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G
    
    input = np.random.rand(4,3,256,192)
    input = torch.Tensor(input)
    out = gram(input)
    print(out.shape) # [4,3,3]
    print(20*'=')
    vggloss = VGGLoss('../vgg_model/vgg19_bn-c79401a0.pth', False)
    pred = torch.rand(1,3,256,192).cuda()
    target = torch.rand(1,3,256,192).cuda()
    out = vggloss(pred,target,None,False,False,True)
    print(out)





    



        






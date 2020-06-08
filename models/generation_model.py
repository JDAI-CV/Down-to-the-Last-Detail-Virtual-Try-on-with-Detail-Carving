import numpy as np
import torch
import os
from .base_model import BaseModel
from models.networks import Define_G, Define_D
from utils.transforms import create_part
import torch.nn.functional as F
from utils import pose_utils
from lib.geometric_matching_multi_gpu import GMM
from .base_model import BaseModel
from time import time
from utils import pose_utils
import os.path as osp
from torchvision import utils
import random

class GenerationModel(BaseModel):
    def name(self):
        return 'Generation model: pix2pix | pix2pixHD'
    def __init__(self, opt):
        self.t0 = time()
        BaseModel.__init__(self, opt)
        self.train_mode = opt.train_mode
        # resume of networks 
        resume_gmm = opt.resume_gmm
        resume_G_parse = opt.resume_G_parse
        resume_D_parse = opt.resume_D_parse
        resume_G_appearance = opt.resume_G_app
        resume_D_appearance = opt.resume_D_app
        resume_G_face = opt.resume_G_face
        resume_D_face = opt.resume_D_face
        # define network
        self.gmm_model = torch.nn.DataParallel(GMM(opt)).cuda()
        self.generator_parsing = Define_G(opt.input_nc_G_parsing, opt.output_nc_parsing, opt.ndf, opt.netG_parsing, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.discriminator_parsing = Define_D(opt.input_nc_D_parsing, opt.ndf, opt.netD_parsing, opt.n_layers_D, 
                                        opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)

        self.generator_appearance = Define_G(opt.input_nc_G_app, opt.output_nc_app, opt.ndf, opt.netG_app, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, with_tanh=False)
        self.discriminator_appearance = Define_D(opt.input_nc_D_app, opt.ndf, opt.netD_app, opt.n_layers_D, 
                                        opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
        
        self.generator_face = Define_G(opt.input_nc_D_face, opt.output_nc_face, opt.ndf, opt.netG_face, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.discriminator_face = Define_D(opt.input_nc_D_face, opt.ndf, opt.netD_face, opt.n_layers_D, 
                                        opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
        
        if opt.train_mode == 'gmm':
            setattr(self, 'generator', self.gmm_model)
        else:
            setattr(self, 'generator', getattr(self, 'generator_' + self.train_mode))
            setattr(self, 'discriminator', getattr(self, 'discriminator_' + self.train_mode))

        # load networks
        self.networks_name = ['gmm', 'parsing', 'parsing', 'appearance', 'appearance', 'face', 'face']
        self.networks_model = [self.gmm_model, self.generator_parsing, self.discriminator_parsing, self.generator_appearance, self.discriminator_appearance, 
                        self.generator_face, self.discriminator_face]
        
        self.networks = dict(zip(self.networks_name, self.networks_model))

        self.resume_path = [resume_gmm, resume_G_parse, resume_D_parse, resume_G_appearance, resume_D_appearance, resume_G_face, resume_D_face]
        for network, resume in zip(self.networks_model, self.resume_path):
            if network != [] and resume != '':
                assert(osp.exists(resume), 'the resume not exits')
                print('loading...')
                self.load_network(network, resume, ifprint=False)

        # define optimizer
        self.optimizer_gmm = torch.optim.Adam(self.gmm_model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        self.optimizer_parsing_G = torch.optim.Adam(self.generator_parsing.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_parsing_D = torch.optim.Adam(self.discriminator_parsing.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        
        self.optimizer_appearance_G = torch.optim.Adam(self.generator_appearance.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_appearance_D = torch.optim.Adam(self.discriminator_appearance.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])

        self.optimizer_face_G = torch.optim.Adam(self.generator_face.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_face_D = torch.optim.Adam(self.discriminator_face.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])

        if opt.train_mode == 'gmm':
            self.optimizer_G = self.optimizer_gmm
        
        elif opt.joint_all:
            self.optimizer_G = [self.optimizer_parsing_G, self.optimizer_appearance_G, self.optimizer_face_G]
            setattr(self, 'optimizer_D', getattr(self, 'optimizer_' + self.train_mode + '_D'))
        
        else:
            setattr(self, 'optimizer_G', getattr(self, 'optimizer_' + self.train_mode + '_G'))
            setattr(self, 'optimizer_D', getattr(self, 'optimizer_' + self.train_mode + '_D'))

        self.t1 = time()

    def set_input(self, opt, result):

        self.t2 = time()
        
        self.source_pose_embedding = result['source_pose_embedding'].float().cuda()
        self.target_pose_embedding = result['target_pose_embedding'].float().cuda()
        self.source_image = result['source_image'].float().cuda()
        self.target_image = result['target_image'].float().cuda()
        self.source_parse = result['source_parse'].float().cuda()
        self.target_parse = result['target_parse'].float().cuda()
        self.cloth_image = result['cloth_image'].float().cuda()
        self.cloth_parse = result['cloth_parse'].float().cuda()
        self.warped_cloth = result['warped_cloth_image'].float().cuda() # preprocess warped image from gmm model
        self.target_parse_cloth = result['target_parse_cloth'].float().cuda()
        self.target_pose_img = result['target_pose_img'].float().cuda()
        self.image_without_cloth = create_part(self.source_image, self.source_parse, 'image_without_cloth', False)
        
        self.im_c = result['im_c'].float().cuda() # target warped cloth

        index = [x for x in list(range(20)) if x != 5 and x != 6 and x != 7]
        real_s_ = torch.index_select(self.source_parse, 1, torch.tensor(index).cuda())
        self.input_parsing = torch.cat((real_s_, self.target_pose_embedding, self.cloth_parse), 1).cuda()
        
        if opt.train_mode == 'gmm':
            self.im_h = result['im_h'].float().cuda()
            self.source_parse_shape = result['source_parse_shape'].float().cuda()
            self.agnostic = torch.cat((self.source_parse_shape, self.im_h, self.target_pose_embedding), dim=1)
                
        elif opt.train_mode == 'parsing':
            self.real_s = self.input_parsing
            self.source_parse_vis = result['source_parse_vis'].float().cuda()
            self.target_parse_vis = result['target_parse_vis'].float().cuda()
    
        elif opt.train_mode == 'appearance':

            if opt.joint_all:
                self.generated_parsing = F.softmax(self.generator_parsing(self.input_parsing), 1)
            else:    
                with torch.no_grad():          
                    self.generated_parsing = F.softmax(self.generator_parsing(self.input_parsing), 1)
            self.input_appearance = torch.cat((self.image_without_cloth, self.warped_cloth, self.generated_parsing), 1).cuda()            
            
            "attention please"
            generated_parsing_ = torch.argmax(self.generated_parsing, 1, keepdim=True)            
            self.generated_parsing_argmax = torch.Tensor()

            for _ in range(20):
                self.generated_parsing_argmax = torch.cat([self.generated_parsing_argmax.float().cuda(), (generated_parsing_ == _).float()], dim=1)
            self.warped_cloth_parse = ((generated_parsing_ == 5) + (generated_parsing_ == 6) + (generated_parsing_ == 7)).float().cuda()

            if opt.save_time:
                self.generated_parsing_vis = torch.Tensor([0]).expand_as(self.target_image)
            else:
                # decode labels cost much time
                _generated_parsing = torch.argmax(self.generated_parsing, 1, keepdim=True)
                _generated_parsing = _generated_parsing.permute(0,2,3,1).contiguous().int()
                self.generated_parsing_vis = pose_utils.decode_labels(_generated_parsing) #array
            
            self.real_s = self.source_image
        
        elif opt.train_mode == 'face':
            if opt.joint_all:# opt.joint
                generated_parsing = F.softmax(self.generator_parsing(self.input_parsing), 1)
                self.generated_parsing_face = F.softmax(self.generator_parsing(self.input_parsing), 1)
            else:
                generated_parsing = F.softmax(self.generator_parsing(self.input_parsing), 1)

                "attention please"
                generated_parsing_ = torch.argmax(generated_parsing, 1, keepdim=True)            
                self.generated_parsing_argmax = torch.Tensor()

                for _ in range(20):
                    self.generated_parsing_argmax = torch.cat([self.generated_parsing_argmax.float().cuda(), (generated_parsing_ == _).float()], dim=1)
                                
                # self.generated_parsing_face = generated_parsing_c
                self.generated_parsing_face = self.target_parse
            
            self.input_appearance = torch.cat((self.image_without_cloth, self.warped_cloth, generated_parsing), 1).cuda()
    
            with torch.no_grad():
                self.generated_inter = self.generator_appearance(self.input_appearance)
                p_rendered, m_composite = torch.split(self.generated_inter, 3, 1) 
                p_rendered = F.tanh(p_rendered)
                m_composite = F.sigmoid(m_composite)
                self.generated_image = self.warped_cloth * m_composite + p_rendered * (1 - m_composite)
            
            self.source_face = create_part(self.source_image, self.source_parse, 'face', False)
            self.target_face_real = create_part(self.target_image, self.generated_parsing_face, 'face', False)
            self.target_face_fake = create_part(self.generated_image, self.generated_parsing_face, 'face', False)
            self.generated_image_without_face = self.generated_image - self.target_face_fake
        
            self.input_face = torch.cat((self.source_face, self.target_face_fake), 1).cuda()
            self.real_s = self.source_face

        elif opt.train_mode == 'joint':
            self.input_joint = torch.cat((self.image_without_cloth, self.warped_cloth, self.generated_parsing), 1).cuda()
    
        self.t3 = time()

        # setattr(self, 'input', getattr(self, 'input_' + self.train_mode))

    def forward(self, opt):
        self.t4 = time()

        if self.train_mode == 'gmm':
            self.grid, self.theta = self.gmm_model(self.agnostic, self.cloth_image)
            self.warped_cloth_predict = F.grid_sample(self.cloth_image, self.grid)

        if opt.train_mode == 'parsing':
            self.fake_t = F.softmax(self.generator_parsing(self.input_parsing), dim=1)
            self.real_t = self.target_parse
        
        if opt.train_mode == 'appearance':
            generated_inter = self.generator_appearance(self.input_appearance)
            p_rendered, m_composite = torch.split(generated_inter, 3, 1) 
            p_rendered = F.tanh(p_rendered)
            self.m_composite = F.sigmoid(m_composite)
            p_tryon = self.warped_cloth * self.m_composite + p_rendered * (1 - self.m_composite)
            self.fake_t = p_tryon
            self.real_t = self.target_image

            if opt.joint_all:

                generate_face = create_part(self.fake_t, self.generated_parsing_argmax, 'face', False)
                generate_image_without_face = self.fake_t - generate_face

                real_s_face = create_part(self.source_image, self.source_parse, 'face', False)
                real_t_face = create_part(self.target_image, self.generated_parsing_argmax, 'face', False)
                input = torch.cat((real_s_face, generate_face), dim=1)

                fake_t_face = self.generator_face(input)
                ###residual learning
                r"""attention
                """
                # fake_t_face = create_part(fake_t_face, self.generated_parsing, 'face', False)
                # fake_t_face = generate_face + fake_t_face
                fake_t_face = create_part(fake_t_face, self.generated_parsing_argmax, 'face', False)
                ### fake image
                self.fake_t = generate_image_without_face + fake_t_face

        if opt.train_mode == 'face':
            self.fake_t = self.generator_face(self.input_face)
            
            if opt.face_residual:
                self.fake_t = create_part(self.fake_t, self.generated_parsing_face, 'face', False)
                self.fake_t = self.target_face_fake + self.fake_t
            
            self.fake_t = create_part(self.fake_t, self.generated_parsing_face, 'face', False)
            self.refined_image = self.generated_image_without_face + self.fake_t
            self.real_t = create_part(self.target_image, self.generated_parsing_face, 'face', False)

        self.t5 = time()

    def backward_G(self, opt):
        self.t6 = time()
                
        if opt.train_mode == 'gmm':
            self.loss = self.criterionL1(self.warped_cloth_predict, self.im_c)
            self.loss.backward()
            self.t7 = time()
            return
        
        fake_st = torch.cat((self.real_s, self.fake_t), 1)
        pred_fake = self.discriminator(fake_st)

        
        if opt.train_mode == 'parsing':
            self.loss_G_GAN = self.criterionGAN(pred_fake,True)
            self.loss_G_BCE = self.criterionBCE_re(self.fake_t, self.real_t) * opt.lambda_L1

            self.loss_G = self.loss_G_GAN + self.loss_G_BCE
            self.loss_G.backward()
        
        if opt.train_mode == 'appearance':          
            self.loss_G_GAN = self.criterionGAN(pred_fake,True) * opt.G_GAN
            # vgg_loss
            loss_vgg1,_ = self.criterion_vgg(self.fake_t, self.real_t, self.target_parse, False, True, False)
            loss_vgg2,_ = self.criterion_vgg(self.fake_t, self.real_t, self.target_parse, False, False, False)
            self.loss_G_vgg = (loss_vgg1 + loss_vgg2) * opt.G_VGG
            self.loss_G_mask = self.criterionL1(self.m_composite, self.warped_cloth_parse) * opt.mask
            if opt.mask_tvloss:
                self.loss_G_mask_tv = self.criterion_tv(self.m_composite)
            else:
                self.loss_G_mask_tv = torch.Tensor([0]).cuda()
            self.loss_G_L1 = self.criterion_smooth_L1(self.fake_t, self.real_t) * opt.lambda_L1

            if opt.joint_all and opt.joint_parse_loss:
                self.loss_G_parsing = self.criterionBCE_re(self.generated_parsing, self.target_parse) * opt.joint_G_parsing
                self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_vgg + self.loss_G_mask + self.loss_G_parsing
            else:
                self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_vgg + self.loss_G_mask + self.loss_G_mask_tv
            self.loss_G.backward()

        if opt.train_mode == 'face':
            _, self.loss_G_vgg = self.criterion_vgg(self.fake_t, self.real_t, self.generated_parsing_face, False, False, False) # part, gram, neareast
            self.loss_G_vgg = self.loss_G_vgg * opt.face_vgg
            self.loss_G_L1 = self.criterionL1(self.fake_t, self.real_t) * opt.face_L1
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * opt.face_gan
            self.loss_G_refine = self.criterionL1(self.refined_image, self.target_image) * opt.face_img_L1

            self.loss_G = self.loss_G_vgg + self.loss_G_L1 + self.loss_G_GAN + self.loss_G_refine
            self.loss_G.backward()

        self.t7 = time()

    def backward_D(self, opt):
        self.t8 = time()

        fake_st = torch.cat((self.real_s, self.fake_t), 1)
        real_st = torch.cat((self.real_s, self.real_t), 1)
        pred_fake = self.discriminator(fake_st.detach())
        pred_real = self.discriminator(real_st) # batch_size,1, 30,30
        
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
    
        self.loss_D.backward()

        self.t9 = time()

    def optimize_parameters(self, opt):
        
        self.t10 = time()
        self.forward(opt)                   # compute fake images: G(A)

        if opt.train_mode == 'gmm':
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G(opt)                  # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
            self.t11 = time()
            return
        
        # update D
        self.set_requires_grad(self.discriminator, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D(opt)                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.discriminator, False)  # D requires no gradients when optimizing G
        if opt.joint_all:
            for _ in self.optimizer_G:
                _.zero_grad()
            
            self.backward_G(opt)

            for _ in self.optimizer_G:
                _.step()
        else:
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G(opt)                  # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights

        self.t11 = time()

    def save_result(self, opt, epoch, iteration):        
        if opt.train_mode == 'gmm':
            images = [self.cloth_image,self.warped_cloth.detach(), self.im_c]

        if opt.train_mode == 'parsing':
            fake_t_vis = pose_utils.decode_labels(torch.argmax(self.fake_t, dim=1, keepdim=True).permute(0,2,3,1).contiguous())
            images = [self.source_parse_vis, self.target_parse_vis, self.target_pose_img, self.cloth_parse, fake_t_vis]

        if opt.train_mode == 'appearance':
            images = [self.image_without_cloth, self.warped_cloth, self.warped_cloth_parse, self.target_image, 
                        self.cloth_image, self.generated_parsing_vis, self.fake_t.detach()]

        if opt.train_mode == 'face':
            images = [self.generated_image.detach(), self.refined_image.detach(), self.source_image, self.target_image, self.real_t, self.fake_t.detach()]


        pose_utils.save_img(images, os.path.join(self.vis_path, str(epoch) + '_' + str(iteration) + '.jpg'))

    def save_model(self, opt, epoch):
        if opt.train_mode == 'gmm':
            model_G = osp.join(self.save_dir, 'generator', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss))
            
            if not osp.exists(osp.join(self.save_dir, 'generator')):
                os.makedirs(osp.join(self.save_dir, 'generator'))

            torch.save(self.generator.state_dict(), model_G)
        
        elif not opt.joint_all:  
            model_G = osp.join(self.save_dir, 'generator', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
            model_D = osp.join(self.save_dir, 'dicriminator', 'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_D))
            if not osp.exists(osp.join(self.save_dir, 'generator')):
                os.makedirs(osp.join(self.save_dir, 'generator'))
            if not osp.exists(osp.join(self.save_dir, 'dicriminator')):
                os.makedirs(osp.join(self.save_dir, 'dicriminator'))
        
            torch.save(self.generator.state_dict(), model_G)
            torch.save(self.discriminator.state_dict(), model_D)
        else:
            model_G_parsing = osp.join(self.save_dir, 'generator_parsing', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
            model_D_parsing = osp.join(self.save_dir, 'dicriminator_parsing', 'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_D))

            model_G_appearance = osp.join(self.save_dir, 'generator_appearance', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
            model_D_appearance = osp.join(self.save_dir, 'dicriminator_appearance', 'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_D))

            model_G_face = osp.join(self.save_dir, 'generator_face', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
            model_D_face = osp.join(self.save_dir, 'dicriminator_face', 'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_D))

            joint_save_dirs = [osp.join(self.save_dir, 'generator_parsing'), osp.join(self.save_dir, 'dicriminator_parsing'),
                                osp.join(self.save_dir, 'generator_appearance'), osp.join(self.save_dir, 'dicriminator_appearance'),
                                osp.join(self.save_dir, 'generator_face'), osp.join(self.save_dir, 'dicriminator_face')]
            for _ in joint_save_dirs:
                if not osp.exists(_):
                    os.makedirs(_)            
            torch.save(self.generator_parsing.state_dict(), model_G_parsing)
            torch.save(self.generator_appearance.state_dict(), model_G_appearance)
            torch.save(self.generator_face.state_dict(), model_G_face)
            torch.save(self.discriminator_appearance.state_dict(), model_D_appearance)
       
    def print_current_errors(self, opt, epoch, i):
        if opt.train_mode == 'gmm':
            errors = {'loss_L1': self.loss.item()}

        if opt.train_mode == 'appearance':
            errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_vgg':self.loss_G_vgg.item(), 'loss_G_mask':self.loss_G_mask.item(),
                        'loss_G_L1': self.loss_G_L1.item(), 'loss_D':self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake':self.loss_D_real.item(), 'loss_G_mask_tv': self.loss_G_mask_tv.item()}
            
            if opt.joint_all and opt.joint_parse_loss:
                errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_vgg':self.loss_G_vgg.item(), 'loss_G_mask':self.loss_G_mask.item(),
                        'loss_G_L1': self.loss_G_L1.item(), 'loss_D':self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake':self.loss_D_real.item(), 'loss_G_parsing': self.loss_G_parsing.item()}


        if opt.train_mode == 'parsing':
            errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_BCE': self.loss_G_BCE.item(), 
                    'loss_D':self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake':self.loss_D_real.item()}

        if opt.train_mode == 'face':
            errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_vgg':self.loss_G_vgg.item(), 'loss_G_refine':self.loss_G_refine.item(),
                        'loss_G_L1': self.loss_G_L1.item(), 'loss_D':self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake':self.loss_D_real.item()}
        
        t = self.t11 - self.t2
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in sorted(errors.items()):
            if v != 0:
                message += '%s: %.3f ' % (k, v)
        
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            



        

        

    


 

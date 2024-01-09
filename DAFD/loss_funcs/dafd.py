from loss_funcs.adv import *
from loss_funcs.lmmd import *
from loss_funcs.adv import *
from loss_funcs.pixel_adv import *
import torch.nn.functional as F
class DAFDLoss(AdversarialLoss, LambdaSheduler):
    def __init__(self, num_class, gamma=1.0, max_iter=1000, recon_lamb=1,lmmd_lamb=1,mi_lamb = 1,adv_lamb=1,dynamic=True,d_lamb =1,pixel_adv_lamb=1,**kwargs):
        super(DAFDLoss, self).__init__(gamma=gamma, max_iter=max_iter, **kwargs)
        self.num_class = num_class
        self.mine = Mine()
        self.recon = Recon()
        self.local_classifiers = torch.nn.ModuleList()
        for _ in range(num_class):
            self.local_classifiers.append(Discriminator())

        self.d_g, self.d_l = 0, 0
        self.dynamic_factor = 0.5
        #adv
        self.adv = AdversarialLoss(gamma=1.0, max_iter=max_iter, use_lambda_scheduler=True, **kwargs)
        #pixel_adv
        self.pixel_adv = Pixel_AdversarialLoss(gamma=1.0, max_iter=max_iter, use_lambda_scheduler=True, **kwargs)
        #lmmd
        self.lmmd = LMMDLoss(num_class=num_class,max_iter=max_iter,dynamic=dynamic,d_lamb = d_lamb,**kwargs)
        #lamb
        self.recon_lamb = recon_lamb
        self.lmmd_lamb = lmmd_lamb
        self.mi_lamb = mi_lamb
        self.adv_lamb = adv_lamb
        self.pixel_adv_lamb = pixel_adv_lamb
    def forward(self, source, target, source_invariant,source_specific,target_invariant,target_specific,source_logits, target_logits,source_base,target_base,source_label,source_front,target_front):

        lamb = self.lamb()
        self.step()

        #source MI
        source_mutual_invariant = F.avg_pool2d(source_invariant, (7, 7))[:,:,0,0]
        source_mutual_specific = F.avg_pool2d(source_specific, (7, 7))[:,:,0,0]
        source_mutual_invariant_shuffle = torch.index_select(source_mutual_invariant, 0, torch.randperm(source_mutual_invariant.shape[0]).cuda())
        source_mi_loss = self.mutual_information_estimator(source_mutual_specific, source_mutual_invariant, source_mutual_invariant_shuffle)
        source_mi_loss = -1.0 * source_mi_loss

        #source recon
        source_recon = torch.cat((source_invariant,source_specific),1)
        source_recon = self.recon(source_recon)

        #target MI
        target_mutual_invariant = F.avg_pool2d(target_invariant, (7, 7))[:,:,0,0]
        target_mutual_specific = F.avg_pool2d(target_specific, (7, 7))[:,:,0,0]
        target_mutual_invariant_shuffle = torch.index_select(target_mutual_invariant, 0, torch.randperm(target_mutual_invariant.shape[0]).cuda())
        target_mi_loss = self.mutual_information_estimator(target_mutual_specific, target_mutual_invariant, target_mutual_invariant_shuffle)
        target_mi_loss = -1.0 * target_mi_loss
        mi_loss = source_mi_loss+target_mi_loss

        #target recon
        target_recon = torch.cat((target_invariant,target_specific),1)
        target_recon = self.recon(target_recon)
        # print(target_recon.shape)

        #recon loss
        source_recon_loss = self.reconstruct_loss(source_recon,source_base.detach())
        target_recon_loss = self.reconstruct_loss(target_recon,target_base.detach())
        
        #lmmd loss
        lmmd_loss = self.lmmd(source, target, source_label,target_logits)
        #adv loss
        adv_loss = self.adv(source,target)
        #pixel_adv_loss
        pixel_adv_loss = self.pixel_adv(source_front,target_front)

        dafd_loss = self.mi_lamb * mi_loss + \
                    (source_recon_loss + target_recon_loss)* self.recon_lamb + \
                    lmmd_loss * self.lmmd_lamb + \
                    adv_loss * self.adv_lamb +\
                    pixel_adv_loss * self.pixel_adv_lamb
        return dafd_loss

    def update_dynamic_factor(self, epoch_length):
        if self.d_g == 0 and self.d_l == 0:
            self.dynamic_factor = 0.5
        else:
            self.d_g = self.d_g / epoch_length
            self.d_l = self.d_l / epoch_length
            self.dynamic_factor = 1 - self.d_g / (self.d_g + self.d_l)
        self.d_g, self.d_l = 0, 0
    def mutual_information_estimator(self, x, y, y_):
        #print(x.shape,y.shape,y_.shape)#torch.Size([32, 2048])
        joint, marginal = self.mine(x, y), self.mine(x, y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))
    def reconstruct_loss(self,src,tgt):
        return torch.sum((src-tgt)**2) / (src.shape[0]*src.shape[1]*src.shape[2]*src.shape[3])
class Mine(nn.Module):
    def __init__(self):
        super(Mine, self).__init__()
        self.fc1_x = nn.Linear(2048, 512)
        self.fc1_y = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512,1)
    def forward(self, x,y):
        # print((self.fc1_x(x)+self.fc1_y(y)).shape)
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2
class Recon(nn.Module):
    def __init__(self):
        super(Recon, self).__init__()
        self.fc = nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        x = self.fc(x)
        return x
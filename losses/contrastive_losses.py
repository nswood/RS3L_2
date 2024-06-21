import torch
import torch.nn as nn
import torch.nn.functional as F
# from hyp_utils import *
import sys
sys.path.append('../FlatSamples_Dev/HypJets')
sys.path.append('../geoopt')

import geoopt
from geoopt.manifolds.stereographic.math import arsinh, artanh,artan_k

import numpy as np
from itertools import combinations
#import torchsort


class CorrLoss(nn.Module):
    def __init__(self, corr=False,ipow=1,sort_tolerance=1.0,sort_reg='l2'):
        super(CorrLoss, self).__init__()
        self.tolerance = sort_tolerance
        self.reg       = sort_reg
        self.corr      = corr
        self.ipow = ipow 
        
    def spearman(self, pred, target):
        #pred   = torchsort.soft_rank(pred.reshape(1,-1))#,regularization=self.reg,regularization_strength=self.tolerance)
        #target = torchsort.soft_rank(target.reshape(1,-1))#,regularization=self.reg,regularization_strength=self.tolerance)
        pred = pred - pred.mean()
        pred =   (pred / pred.norm())
        target = target - target.mean()
        target = (target / target.norm())
        pred   = pred.pow(self.ipow)
        target = target.pow(self.ipow)
        ret = ((pred * target)).sum()
        if self.corr:
            return (1-ret)*(1-ret)
        else:
            return ret*ret
    
    def forward(self, features, labels,mask=None):
        if mask is not None:
            featuretest=features[mask]
            labeltest=labels[mask]
        else:
            featuretest=features
            labeltest=labels
        return self.spearman(featuretest,labeltest) 


class CorrLoss2(nn.Module):
    def __init__(self,background_only=False,anti=False,background_label=1,power=2):
        self.backonly = background_only
        self.background_label = background_label
        self.power = power
        self.anti  = anti

    def distance_corr(self,var_1,var_2,normedweight,power=1):
        xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
        yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
        amat = (xx-yy).abs()
        del xx,yy

        amatavg = torch.mean(amat*normedweight,dim=1)
        Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))-amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))+torch.mean(amatavg*normedweight)
        del amat
        
        xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
        yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
        bmat = (xx-yy).abs()
        del xx,yy

        bmatavg = torch.mean(bmat*normedweight,dim=1)
        Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
            -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
            +torch.mean(bmatavg*normedweight)
        del bmat

        ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
        AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
        BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)
        del Bmat, Amat

        if(power==1):
            dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
        elif(power==2):
            dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
        else:
            dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power

        return dCorr

    def __call__(self,pred,x_biased,weights=None):
        xweights = torch.ones_like(pred)
        disco = self.distance_corr(x_biased,pred,normedweight=xweights,power=self.power)
        if self.anti:
            disco = 1-disco
        return disco


class SimCLRLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward2(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        features = features.squeeze()
        dist = (-features.unsqueeze(0)+features.unsqueeze(0).transpose(0, 1)).norm(dim = -1, p = 2,keepdim = False)
        dist = 1/(1+dist)
        anchor_dot_contrast = torch.div(
            dist,
            self.temperature)
#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.T),
#             self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
#         print(f"exp_logits:{torch.isnan(exp_logits).any()}" )
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
    
    def forward(self, x_i, x_j):
        #xdevice = x_i.get_device()
        xdevice = (torch.device('cuda') if x_i.is_cuda else torch.device('cpu'))
        batch_size = x_i.shape[0]
        z_i = F.normalize( x_i, dim=1 )
        z_j = F.normalize( x_j, dim=1 )
        z   = torch.cat( [z_i, z_j], dim=0 )
        similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
        sim_ij = torch.diag( similarity_matrix,  batch_size )
        sim_ji = torch.diag( similarity_matrix, -batch_size )
        positives = torch.cat( [sim_ij, sim_ji], dim=0 )
        nominator = torch.exp( positives / self.temperature )
        negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float()
        negatives_mask = negatives_mask.to( xdevice )
        denominator = negatives_mask * torch.exp( similarity_matrix / self.temperature )
        loss_partial = -torch.log( nominator / torch.sum( denominator, dim=1 ) )
        loss = torch.sum( loss_partial )/( 2*batch_size )
        return loss

class VICRegLoss(torch.nn.Module):

    def __init__(self, lambda_param=1,mu_param=1,nu_param=20,sort_tolerance=1.0,sort_reg='l2'):
        super(VICRegLoss, self).__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.tolerance = sort_tolerance
        self.reg       = sort_reg

    def forward(self, x, y):
        #self.device = (torch.device('cuda') if x.is_cuda else torch.device('cpu'))
        
        #x_scale = x
        #y_scale = y
        repr_loss = F.mse_loss(x, y)
        
        #x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        N = x.size(0)
        D = x.size(1)
        
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        #x = torchsort.soft_rank(x.cpu(),regularization=self.reg,regularization_strength=self.tolerance,)
        #y = torchsort.soft_rank(y.cpu(),regularization=self.reg,regularization_strength=self.tolerance,)
        #x = x.cuda()
        #y = y.cuda()
        x = (x-x.mean(dim=0))/x.std(dim=0)
        y = (y-y.mean(dim=0))/y.std(dim=0)
        #x_scale = x_scale/x_scale.norm()
        #y_scale = y_scale/y_scale.norm()
        #z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        #z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        cov_x = (x.T @ x) / (N - 1)
        cov_y = (y.T @ y) / (N - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(D) + self.off_diagonal(cov_y).pow_(2).sum().div(D)
        return repr_loss,cov_loss,std_loss

    def off_diagonal(self,x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

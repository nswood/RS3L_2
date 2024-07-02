import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn
import torch.utils.data
from torch.autograd.variable import *
import torch.optim as optim
import torch.nn.functional as F
import datetime
import os
import numpy as np
import h5py
import json
import sklearn
import numpy.random as random
from utils import utils 
import sys
import glob
import losses
import itertools
import matplotlib.pyplot as plt
import sys


p = utils.ArgumentParser()
p.add_args(
    ('--mname', p.STR),
    
    # Paths to data
    ('--ipath', p.STR), ('--vpath', p.STR), ('--opath', p.STR), 
    
    # CLR hyperparams
    ('--temperature', p.FLOAT), 
    
    # Training parameters
    ('--nepochs', p.INT),('--batchsize', p.INT),('--dropout_rate', p.FLOAT),('--lr', {'type': float}),('--weight_init', {'type': float}),
    
    # Data Characteristics
    ('--num_max_particles', p.INT),('--num_max_files', p.INT),
    ('--nparts', p.INT),('--nclasses',p.INT),
    ('--part_feature_size', p.INT),('--jet_feature_size', p.INT),#('--feature_sv_size', p.INT),
    
    # Continue training
    ('--mpath', p.STR), ('--continue_training', p.STORE_TRUE),
    
    # Pretrained model
    ('--pretrain', p.STORE_TRUE), ('--prepath', p.STR),
    
    # Models
    ('--trans', p.STORE_TRUE),('--dnn', p.STORE_TRUE),('--gnv2', p.STORE_TRUE),
    ('--gnv2_no_sv', p.STORE_TRUE),('--pnet', p.STORE_TRUE),('--pnet_tagger', p.STORE_TRUE),
    
    # Model parameters
    ('--dropout', {'type': float}),
    # Transformer specific parameters
    ('--class_att', p.STORE_TRUE),('--num_encoders', p.INT),('--num_attention_heads', p.INT),('--n_out_nodes',p.INT),
    
    # Graph specific parameters
    ('--hidden_size',p.INT),('--De',p.INT),('--Do',p.INT),('--graph_attention',p.STORE_TRUE),('--event_branch',p.STORE_TRUE),
    )

#DDP Configs
p.add_argument('--gpu', default=None, type=int)
p.add_argument('--device', default='cuda', help='device')
p.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
p.add_argument('--rank', default=-1, type=int, 
                    help='node rank for distributed training')
p.add_argument('--dist-url', default='env://', type=str, 
                    help='url used to set up distributed training')
p.add_argument('--dist-backend', default='nccl', type=str, 
                    help='distributed backend')
p.add_argument('--local_rank', default=-1, type=int, 
                    help='local rank for distributed training')
p.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
args = p.parse_args()

# args.nparts = 100
np.random.seed(42)


from utils.Trainer import Trainer

def ddp_setup():
    if 'SLURM_PROCID' in os.environ: # for slurm scheduler
        args.rank = int(os.environ['SLURM_PROCID'])
        print(args.rank)
        args.gpu = args.rank % torch.cuda.device_count()
        print("Available GPUs:", torch.cuda.device_count())
        print("LOCAL_RANK:", args.gpu)
    else:
        print("Available GPUs:", torch.cuda.device_count())
        print("LOCAL_RANK:", int(os.environ["LOCAL_RANK"]))
    os.environ['RANK'] = str(args.rank)
    init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)



def load_data():
    from dataloaders.all_data_loader import zpr_loader
    
    '''
    
    Update with path to data
    
    '''
    
    data_train = zpr_loader(None,None,maxfiles=args.num_max_files,pf_size = args.feature_size, max_num_particles = args.nparts)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, shuffle=True)
    train_loader = DataLoader(data_train, batch_size=args.batchsize,shuffle=(train_sampler is None),
        sampler=train_sampler)
    
    
    
    data_val = zpr_loader(None,None,maxfiles=args.num_max_files,equal_qcd=args.equal_qcd, max_num_particles = args.nparts)
    val_sampler = torch.utils.data.distributed.DistributedSampler(data_val, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=args.batchsize,shuffle=(val_sampler is None),
         sampler=val_sampler)
    
    return train_loader, val_loader
    
def load_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    if args.trans:
        from models.transformer import trans
        model = trans(args, args.mname, _softmax, _sigmoid, pretrain=args.pretrain)
    elif args.gnv2:
        from models.graph_models import GraphNetv2
        model = GraphNetv2(
            name=args.mname,
            n_constituents=args.num_max_particles,
            n_targets=args.nclasses,
            params=args.part_feature_size,
#             params_v=args.feature_sv_size,
            event_branch=args.event_branch,
            hidden=args.hidden_size,
            De=args.De,
            Do=args.Do,
            dropout=args.dropout_rate,
            softmax=_softmax,
            sigmoid=_sigmoid,
            attention_flag=args.graph_attention
        )
    elif args.gnv2_no_sv:
        from models.graph_models import GraphNetnoSV
        model = GraphNetnoSV(
            name=args.mname,
            n_constituents=args.num_max_particles,
            n_targets=args.nclasses,
            params=args.part_feature_size,
            hidden=args.hidden_size,
            De=args.De,
            Do=args.Do,
            softmax=_softmax
        )
    elif args.pnet:
        from models.graph_models import ParticleNet
        model = ParticleNet(
            input_dims=args.part_feature_size,
            num_classes=args.nclasses,
            conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
            fc_params=[(128, 0.1)],
            use_fusion=True,
            use_fts_bn=True,
            use_counts=True,
            for_inference=False,
            sigmoid=_sigmoid
        )
    elif args.pnet_tagger:
        from models.graph_models import ParticleNetTagger
        model = ParticleNetTagger(
            name=args.mname,
            pf_features_dims=args.part_feature_size,
#             sv_features_dims=args.feature_sv_size,
            num_classes=args.nclasses,
            conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
            fc_params=[(128, 0.1)],
            use_fusion=True,
            use_fts_bn=True,
            use_counts=True,
            for_inference=False,
            sigmoid=_sigmoid
        )
    elif args.dnn:
        from models.dnn import DNN
        model = DNN(
            name=args.mname,
            n_inputs=args.part_feature_size,
            n_targets=args.nclasses
        )
            
    num_params = count_parameters(model)
    return model,num_params



import geoopt
from lion_pytorch.Rlion_pytorch import *
from lion_pytorch.lion_pytorch import Lion
from torch.optim.lr_scheduler import OneCycleLR

def load_train_objs():
    model,num_params = load_model()
    
    if args.fix_weights:
        print('Freezing Weights')
        for name, param in model.named_parameters():
            #Unfreeze final layers    

            if not ('final_embedder' in name): 
                print(name)
                param.requires_grad = False
    
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay = 5e-4,betas=(0.9, 0.98))
   
    eta_min = args.lr/10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.nepochs, eta_min=eta_min)
    
    return model,num_params, optimizer,scheduler


def main(save_every: int, total_epochs: int, batch_size: int):
    ddp_setup()
    torch.set_default_dtype(torch.float64)
    
    model,num_params, optimizer,scheduler = load_train_objs()
    model.double()
    
#     train_loader, val_loader = load_data()
    outdir = f"./{args.opath}/{model.name.replace(' ','_')}"

    outdir = utils.makedir(outdir,args.continue_training)

    with open(os.path.join(outdir,'params.txt'), "w") as file:
        file.write(str(num_params))

    print(f"Number of parameters in the model: {num_params}")
    with open(os.path.join(outdir,'train_size.txt'), "w") as file:
        file.write(str(len(train_loader.dataset)))
    
#     trainer = Trainer(model, train_loader,val_loader, optimizer, save_every,outdir,total_epochs, args,scheduler)
    trainer = Trainer(model, None,None, optimizer, save_every,outdir,total_epochs, args,scheduler)

    trainer.train(total_epochs)
  
if __name__ == "__main__":
    n_particles = args.nparts
    
    _sigmoid=False
    _softmax=False
    main(25, args.nepochs, args.batchsize)
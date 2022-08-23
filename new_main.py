import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import grad 

import numpy as np
import random
import wandb
from tqdm import tqdm
from PIL import Image
from packaging import version
from datetime import datetime

# models
from model.refinenetlw import rf_lw101
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1

# datasets
#from dataset.paired_cityscapes import Pairedcityscapes
#from dataset.Foggy_Zurich import foggyzurichDataSet
from dataset.rgbdataset import RGBDataset
from dataset.thermaldataset import ThermalDataset

# utils
from configs.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers
from utils.losses import CrossEntropy2d

# metrics
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer

IMG_MEAN = (0, 0, 0) # won't need this once the dataset is replaced for the right one

def seg_loss(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred, label)

def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type="sgd",
        enc_lr=6e-4,
        enc_weight_decay=1e-5,
        enc_momentum=0.9,
        dec_optim_type="sgd",
        dec_lr=6e-3,
        dec_weight_decay=1e-5,
        dec_momentum=0.9,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=0.5,
        dec_lr_gamma=0.5,
        enc_scheduler_type="multistep",
        dec_scheduler_type="multistep",
        epochs_per_stage=(100, 100, 100),
    )
    return optimisers, schedulers

def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

def main():
    args = get_arguments()

    # set up random seeds
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # name the log file
    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-{now}'

    wandb.init(project='FIFO',name=f'{run_name}')
    wandb.config.update(args)

    cudnn.enabled = True
    
    # get segmentation model
    start_iter = 0
    model = rf_lw101(num_classes=args.num_classes)

    model.train()
    model.cuda(args.gpu)

    # learning rates
    lr_df1 = 1e-3 
    lr_df2 = 1e-3

    if args.modeltrain=='train':
        lr_df1 = 5e-4
    
    DomainFilter1 = FogPassFilter_conv1(2080)
    DomainFilter1_optimizer = torch.optim.Adamax([p for p in DomainFilter1.parameters() if p.requires_grad == True], lr=lr_df1)
    DomainFilter1.cuda(args.gpu)

    DomainFilter2 = FogPassFilter_res1(32896)
    DomainFilter2_optimizer = torch.optim.Adamax([p for p in DomainFilter2.parameters() if p.requires_grad == True], lr=lr_df2)
    DomainFilter2.cuda(args.gpu)

    domainfilter_loss = losses.ContrastiveLoss(
        pos_margin=0.1,
        neg_margin=0.1,
        distance=CosineSimilarity(),
        reducer=MeanReducer()
        )
    
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # load thermal/rgb datasets separately for the segmenation training and the domain filter training

    # thermal_loader_seg = data.DataLoader(Pairedcityscapes(args.data_dir, args.data_dir_cwsf, args.data_list, args.data_list_cwsf,
    #                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                                     mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                                     pin_memory=True)
    
    # rgb_loader_seg = data.DataLoader(Pairedcityscapes(args.data_dir, args.data_dir_cwsf, args.data_list, args.data_list_cwsf,
    #                                             max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                                             mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                                             pin_memory=True)

    # thermal_loader_df = data.DataLoader(Pairedcityscapes(args.data_dir, args.data_dir_cwsf, args.data_list, args.data_list_cwsf,
    #                                             max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                                             mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                                             pin_memory=True)
    
    # rgb_loader_df = data.DataLoader(Pairedcityscapes(args.data_dir, args.data_dir_cwsf, args.data_list, args.data_list_cwsf,
    #                                             max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                                             mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                                             pin_memory=True)
    data_root = '../../thermal_data/annotated_thermal_datasets/'
    thermal_loader_seg = data.DataLoader(ThermalDataset(data_root, 'training'), batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    rgb_loader_seg = data.DataLoader(RGBDataset(data_root, 'training'), batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    thermal_loader_df = data.DataLoader(ThermalDataset(data_root, 'training'), batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    rgb_loader_df = data.DataLoader(RGBDataset(data_root, 'training'), batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    thermal_loader_seg_iter = enumerate(thermal_loader_seg)
    rgb_loader_seg_iter = enumerate(rgb_loader_seg)

    thermal_loader_df_iter = enumerate(thermal_loader_df)
    rgb_loader_df_iter = enumerate(rgb_loader_df)

    # load optimizers and schedulers
    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=model)
    opts = make_list(optimisers)

    for i_iter in tqdm(range(start_iter, args.num_steps)):

        loss_seg_thermal_val = 0
        loss_seg_rgb_val = 0
        loss_sm_val = 0

        for opt in opts:
            opt.zero_grad()

        for sub_i in range(args.iter_size):
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            for param in DomainFilter1.parameters():
                param.requires_grad = True
            for param in DomainFilter2.parameters():
                param.requires_grad = True
        
        # get a batch of thermal and rgb images
        _, batch  = thermal_loader_df_iter.__next__
        thermal_image, label_thermal, size = batch
        _, batch  = rgb_loader_df_iter.__next__
        rgb_image, label_rgb, size = batch
        interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')

        # generate features
        thermal_image = Variable(thermal_image).cuda(args.gpu)
        feature_t0, feature_t1, feature_t2, feature_t3, feature_t4, feature_t5 = model(thermal_image)

        rgb_image = Variable(rgb_image).cuda(args.gpu)
        feature_rgb0, feature_rgb1, feature_rgb2, feature_rgb3, feature_rgb4, feature_rgb5 = model(rgb_image)

        fsm_weights = {'layer0':0.5, 'layer1':0.5}
        rgb_features = {'layer0': feature_rgb0, 'layer1': feature_rgb1}
        thermal_features = {'layer0': feature_t0, 'layer1': feature_t1}

        total_df_loss = 0

        for idx, layer in enumerate(fsm_weights):
            # find domain filter loss for both layers

            thermal_feature = thermal_features[layer]
            rgb_feature = rgb_features[layer]

            if idx == 0:
                domain_filter = DomainFilter1
                domain_filter_optimizer = DomainFilter1_optimizer
            elif idx == 1:
                domain_filter = DomainFilter2
                domain_filter_optimizer = DomainFilter2_optimizer

            domain_filter.train()  
            domain_filter_optimizer.zero_grad()

            thermal_gram = [0]*args.batch_size
            rgb_gram = [0]*args.batch_size
            vector_thermal_gram = [0]*args.batch_size
            vector_rgb_gram = [0]*args.batch_size
            thermal_factor = [0]*args.batch_size
            rgb_factor = [0]*args.batch_size

            for batch_idx in range(args.batch_size):
                # make gram matrix, upper diagnol as vector, factor vectors
                thermal_gram = gram_matrix(thermal_feature[batch_idx])
                rgb_gram = gram_matrix(rgb_feature[batch_idx])

                vector_thermal_gram = Variable(thermal_gram[batch_idx][torch.triu(torch.ones(thermal_gram[batch_idx].size()[0], thermal_gram[batch_idx].size()[1])) == 1], requires_grad=True)
                vector_rgb_gram = Variable(rgb_gram[batch_idx][torch.triu(torch.ones(rgb_gram[batch_idx].size()[0], rgb_gram[batch_idx].size()[1])) == 1], requires_grad=True)

                thermal_factor = domain_filter(vector_thermal_gram[batch_idx])
                rgb_factor = domain_filter(vector_rgb_gram[batch_idx])
            
            domain_factor_embeddings = torch.cat((torch.unsqueeze(thermal_factor[0],0),torch.unsqueeze(rgb_factor[0],0),
                                                   torch.unsqueeze(thermal_factor[1],0),torch.unsqueeze(rgb_factor[1],0),
                                                   torch.unsqueeze(thermal_factor[2],0),torch.unsqueeze(rgb_factor[2],0),
                                                   torch.unsqueeze(thermal_factor[3],0),torch.unsqueeze(rgb_factor[3],0)),0)

            domain_factor_embeddings_norm = torch.norm(domain_factor_embeddings, p=2, dim=1).detach()
            size_domain_factor = domain_factor_embeddings.size()
            domain_factor_embeddings = domain_factor_embeddings.div(domain_factor_embeddings_norm.expand(size_domain_factor[1],8).t())
            domain_factor_labels = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1])
            domain_filter_loss_val = domainfilter_loss(domain_factor_embeddings,domain_factor_labels)

            total_df_loss +=  domain_filter_loss_val

            wandb.log({f'layer{idx}/fpf loss': domain_filter_loss_val}, step=i_iter)
            wandb.log({f'layer{idx}/total fpf loss': total_df_loss}, step=i_iter)

        # update total domain filter loss for the batch
        total_df_loss.backward(retain_graph=False)

        if args.modeltrain=='train':
            # train segmentation network
            # freeze the parameters of fog pass filtering modules

            model.train()
            for param in model.parameters():
                param.requires_grad = True
            for param in DomainFilter1.parameters():
                param.requires_grad = False
            for param in DomainFilter2.parameters():
                param.requires_grad = False

            # get a batch of thermal and rgb images
            _, batch  = thermal_loader_seg_iter.__next__
            thermal_image, label_thermal, size = batch
            _, batch  = rgb_loader_seg_iter.__next__
            rgb_image, label_rgb, size = batch
            interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')

            # get features

            thermal_image = Variable(thermal_image).cuda(args.gpu)
            feature_t0, feature_t1, feature_t2, feature_t3, feature_t4, feature_t5 = model(thermal_image)

            rgb_image = Variable(rgb_image).cuda(args.gpu)
            feature_rgb0, feature_rgb1, feature_rgb2, feature_rgb3, feature_rgb4, feature_rgb5 = model(rgb_image)

            fsm_weights = {'layer0':0.5, 'layer1':0.5}
            rgb_features = {'layer0': feature_rgb0, 'layer1': feature_rgb1}
            thermal_features = {'layer0': feature_t0, 'layer1': feature_t1}

            # thermal segmentation loss
            pred_t5 = interp(feature_t5)
            loss_seg_thermal = seg_loss(pred_t5, label_thermal, args.gpu)

            # rgb segmentation loss
            pred_rgb5 = interp(feature_rgb5)
            loss_seg_rgb = seg_loss(pred_rgb5, label_rgb, args.gpu)

            loss_sm = 0
            domain_filter_loss = 0

            for idx, layer in enumerate(fsm_weights):
                
                layer_sm_loss = 0
                domain_filter_loss = 0

                thermal_feature = thermal_features[layer]
                rgb_feature = rgb_features[layer]

                na,da,ha,wa = thermal_feature.size()
                nb,db,hb,wb = rgb_feature.size()

                if idx == 0:
                    domain_filter = DomainFilter1
                    domain_filter_optimizer = DomainFilter1_optimizer
                elif idx == 1:
                    domain_filter = DomainFilter2
                    domain_filter_optimizer = DomainFilter2_optimizer

                domain_filter.eval()

                # calculate style matching loss

                for batch_idx in range(4):
                    thermal_gram = gram_matrix(thermal_feature[batch_idx])
                    rgb_gram = gram_matrix(rgb_feature[batch_idx])

                    vector_thermal_gram = Variable(thermal_gram[batch_idx][torch.triu(torch.ones(thermal_gram[batch_idx].size()[0], thermal_gram[batch_idx].size()[1])) == 1], requires_grad=True)
                    vector_rgb_gram = Variable(rgb_gram[batch_idx][torch.triu(torch.ones(rgb_gram[batch_idx].size()[0], rgb_gram[batch_idx].size()[1])) == 1], requires_grad=True)

                    thermal_factor = domain_filter(vector_thermal_gram[batch_idx])
                    rgb_factor = domain_filter(vector_rgb_gram[batch_idx])

                    half = int(rgb_factor.shape[0]/2)
                    layer_sm_loss += fsm_weights[layer]*torch.mean((rgb_factor/(hb*wb) - thermal_factor/(ha*wa))**2)/half/ rgb_feature.size(0)

                loss_sm += layer_sm_loss / 4.

                loss = loss_sm + loss_seg_thermal + loss_seg_rgb
                loss = loss / args.iter_size
                loss.backward()

                if loss_seg_thermal != 0:
                    loss_seg_thermal_val += loss_seg_thermal.data.cpu().numpy() / args.iter_size
                if loss_seg_rgb != 0:
                    loss_seg_rgb_val += loss_seg_rgb.data.cpu().numpy() / args.iter_size
                if loss_sm != 0:
                    loss_sm_val += loss_sm.data.cpu().numpy() / args.iter_size
            
                wandb.log({"fsm loss": args.lambda_fsm*loss_sm_val}, step=i_iter)
                wandb.log({'thermal_loss_seg': loss_seg_thermal_val}, step=i_iter)
                wandb.log({'rgb_loss_seg': loss_seg_rgb_val}, step=i_iter)
                wandb.log({'total_loss': loss}, step=i_iter)           

                for opt in opts:
                    opt.step()

            DomainFilter1_optimizer.step()
            DomainFilter2_optimizer.step()
        
        # how often to save predictions and models

        if i_iter < 20000:
            save_pred_every = 5000
        if args.modeltrain=='train':
            save_pred_every = 2000
        else:
            save_pred_every = args.save_pred_every

        if i_iter >= args.num_steps_stop - 1:
            print('save model ..')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.file_name + str(args.num_steps_stop) + '.pth'))
            break
        if args.modeltrain != 'train':
            if i_iter == 5000:
                torch.save({'state_dict':model.state_dict(),
                'df1_state_dict':DomainFilter1.state_dict(),
                'df2_state_dict':DomainFilter2.state_dict(),
                'train_iter':i_iter,
                'args':args
                },osp.join(args.snapshot_dir, run_name)+'_domainfilter_'+str(i_iter)+'.pth')

        if i_iter % save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            save_dir = osp.join(f'./result/FIFO_model', args.file_name)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save({
                'state_dict':model.state_dict(),
                'df1_state_dict':DomainFilter1.state_dict(),
                'df2_state_dict':DomainFilter2.state_dict(),
                'train_iter':i_iter,
                'args':args
            },osp.join(args.snapshot_dir, run_name)+'_FIFO'+str(i_iter)+'.pth')
            
if __name__ == '__main__':
    main()

import argparse
import logging
import os
import random
from utils import Miou
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.used_function import assign_superpixel_label_refine, kl_loss_compute
from dataloaders.base_dataset import train_dataset,train_transform,test_transform,train_dataset_noread_distance_jit_SLIC

import warnings
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='debug', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--learning_rate', type=float,  default=0.005, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--noise_type', type=str, default='02', help='clean,02,08,DE')
# noise 02 represent the SR simulated noise, 08 represent the SE simulated noise, DE represent the SDE simulated noise,
parser.add_argument('--clean_part', type=float, default=0.0, help='part_for_clean')
parser.add_argument('--data_rate', type=float, default=1.0, help='part_for_clean')
parser.add_argument('--datasets', type=str,  default='kvasir', help='ISIC,kvasir,shenzhen')
parser.add_argument('--ratecp', type=float,  default=0.5, help='kvasir,shenzhen,BUSUC')
parser.add_argument('--forget_rate', type=float, default=0.15, help='top_num')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from utils.select_dataset import data_path

def train(args, snapshot_path):

    batch_size = args.batch_size

    path_img, path_gt_clean, path_gt, path_img_test, path_gt_test,path_SLIC=data_path(args)  # use this code to identify data path
    names = os.listdir(path_img)#[:50] choose less data for debug
    random.shuffle(names)
    names_use = int(args.data_rate * len(names))
    names = names[:names_use]
    from networks.unet_model import UNet
    model = UNet(3,2).cuda()
    model1 = UNet(3, 2).cuda()

    # define datasets
    imgs = [cv2.imread(path_img+name)[:,:,::-1] for name in names]
    # extract the superpixel Structural prior
    imgs_SLIC = [np.load(path_SLIC + name.replace('.png', '.npy')) for name in names]

    # gts = []
    # for name in names:
    gts = [cv2.imread(path_gt + name.replace('.jpg', '.png'))[:, :, 0] / 255 for name in names]
        # gts.append(gt)
    db_train = train_dataset_noread_distance_jit_SLIC(imgs=imgs, gts = gts,SLIC=imgs_SLIC, transform=train_transform,datasets_h=args.datasets)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn,drop_last=True)

    model.train()
    model1.train()

    optimizer = optim.SGD(list(model.parameters())+list(model1.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

    ce_reduce_none = nn.CrossEntropyLoss(ignore_index=2,reduction='none')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0

    max_epoch = args.max_epochs
    iterator = tqdm(range(max_epoch), ncols=70)

    forget_rate_ori = args.forget_rate
    forget_rate = args.forget_rate
    num_gradual = 20
    rate_schedule = np.ones(100) * forget_rate
    rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** 1, num_gradual)
    co_lambda=0.5
    from utils.losses import SoftDiceLoss
    dice_loss_soft0 = SoftDiceLoss(smooth=1.)
    writer = SummaryWriter(snapshot_path + '/log')
    final_dc = []
    for iter in iterator:
        model.train()
        model1.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            iter_for_use = iter//10
            forget_rate = rate_schedule[iter]
            volume_batch, volume_batch_jit, label_batch, distance_batch,img_SLIC = sampled_batch
            volume_batch, volume_batch_jit, label_batch, distance_batch,img_SLIC = \
                volume_batch.cuda().float(), volume_batch_jit.cuda().float(), label_batch.cuda().long(), distance_batch.cuda().long(),distance_batch.cuda().long()

            distance_batch_10 = torch.clip(distance_batch, 1, max(11 - iter_for_use, 1))

            outputs = model(volume_batch)
            outputs1 = model1(volume_batch_jit)
            loss_pick_1 = F.cross_entropy(outputs, label_batch,reduce = False)
            loss_pick_2 = F.cross_entropy(outputs1, label_batch,reduce = False)
            loss_pick_1_inverse = F.cross_entropy(outputs, 1 - label_batch, reduce=False)
            loss_pick_2_inverse = F.cross_entropy(outputs1, 1 - label_batch, reduce=False)

            loss_pick = (loss_pick_1 + loss_pick_2 + (1-co_lambda) * kl_loss_compute(outputs, outputs1, reduce=False) + (1-co_lambda )* kl_loss_compute(outputs1, outputs, reduce=False))#.cpu()
            loss_pick_sup = loss_pick_1 + loss_pick_2
            loss_pick_for_search_tau = torch.flatten(loss_pick_sup[:, ]).flatten(loss_pick[:, ])
            loss_pick2 = torch.flatten(loss_pick[:, ])
            loss_pick_inverse = (loss_pick_1_inverse + loss_pick_2_inverse)  # .cpu()
            distance10_flat = torch.flatten(distance_batch_10[:, ])
            ind_sorted = torch.argsort(loss_pick2.data)
            loss_sorted = loss_pick2[ind_sorted]
            loss_sorted_search_tau = loss_pick_for_search_tau[ind_sorted]
            loss_pick2_inverse = torch.flatten(loss_pick_inverse[:, ])
            loss_sorted_inverse = loss_pick2_inverse[ind_sorted]
            distancewith_loss_sorted = distance10_flat[ind_sorted]#.cpu()

            remember_rate = 1 - forget_rate
            num_remember = int(remember_rate * len(loss_sorted))
            ind_update = (loss_sorted*distancewith_loss_sorted)[:num_remember]
            loss = torch.mean(ind_update)
            # num_remember=int(0.6* len(loss_sorted))
            for_pl_index = torch.zeros_like(ind_sorted)
            for_pl_index[ind_sorted[:num_remember]]=1    #这个表示有用的都是1
            for_pl_mask = for_pl_index.reshape(args.batch_size,256,256)   # 这里的1表示loss是低的
            num_remember_ori_from_rate = int((1 - forget_rate_ori) * len(loss_sorted_search_tau))
            loss_used_part = loss_sorted_search_tau[:num_remember_ori_from_rate]

            loss_inverst_ignore_part = loss_sorted_inverse[num_remember_ori_from_rate:]
            loss_use_and_ignore_inverse = torch.cat([loss_used_part, loss_inverst_ignore_part], dim=0)
            writer.add_scalar("Loss/loss_use_and_ignore_part_inverse", torch.mean(loss_use_and_ignore_inverse), iter_num)
            if iter < 30:
                loss3 = 0
                loss4 = 0
            else:
                outputs_soft1 = torch.softmax(outputs, dim=1)
                outputs_soft2 = torch.softmax(outputs1, dim=1)
                alpha = random.random() + 1e-5
                outputs_soft = alpha * outputs_soft1 + (1-alpha)*outputs_soft2


                used_ori = for_pl_mask

                masks_generated = torch.zeros_like(label_batch)
                for idxx in range(batch_size):
                    generated_mask_i = assign_superpixel_label_refine(score=outputs_soft[idxx,::], segments=img_SLIC[idxx,::], threshold=0.5)
                    masks_generated[idxx,::] = generated_mask_i

                new_label = masks_generated * (1-used_ori) + used_ori*label_batch   # 这个代码融合了距离大于10的区域，该区域被定义为好区域
                new_label_2 =  torch.where(new_label==2,1,0)
                new_label_1 = torch.where(new_label == 1, 1, 0)
                new_label_use_zero = torch.where(new_label == 0, 1, 0)

                new_label_use = new_label_1

                idxx = np.arange(0, args.batch_size, 1)
                random.shuffle(idxx)
                volume_batch22,label_batch22 = volume_batch[idxx],label_batch[idxx]
                volume_batch_jit22 = volume_batch_jit[idxx]
                new_label_idxx = new_label[idxx]
                weight_map = distance_batch_10.clone()
                weight_map_idxx = distance_batch_10[idxx]

                volume_batch_mixed = torch.zeros_like(volume_batch)
                loss_weighted_mixed = torch.zeros_like(new_label)
                label_batch_mixed = torch.zeros_like(new_label)
                volume_batch_mixed1 = torch.zeros_like(volume_batch)
                label_batch_mixed1 = torch.zeros_like(new_label)
                summs = new_label.sum(dim=[1, 2], keepdim=False)
                for i in range(args.batch_size):
                    rate = summs[i] / 65536  # 256*256
                    if rate > args.ratecp:
                        volume_batch_mixed[i] = volume_batch[i] * new_label_use_zero[i].unsqueeze(0) + (
                                1 - new_label_use_zero[i]).unsqueeze(0) * volume_batch22[i]
                        label_batch_mixed[i] = new_label[i] * new_label_use_zero[i] + (1 - new_label_use_zero[i]) * \
                                               new_label_idxx[i]
                        loss_weighted_mixed[i] = weight_map[i] * new_label_use_zero[i] + (1 - new_label_use_zero[i]) * \
                                                 weight_map_idxx[i]

                        volume_batch_mixed1[i] = volume_batch_jit[i] * new_label_use_zero[i].unsqueeze(0) + (
                                1 - new_label_use_zero[i]).unsqueeze(0) * volume_batch_jit22[i]

                    else:
                        volume_batch_mixed[i] = volume_batch[i] * new_label_use[i].unsqueeze(0) + (
                                1 - new_label_use[i]).unsqueeze(0) * volume_batch22[i]
                        label_batch_mixed[i] = new_label[i] * new_label_use[i] + (1 - new_label_use[i]) * \
                                               new_label_idxx[i]
                        loss_weighted_mixed[i] = weight_map[i] * new_label_use[i] + (1 - new_label_use[i]) * \
                                                 weight_map_idxx[i]

                        volume_batch_mixed1[i] = volume_batch_jit[i] * new_label_use[i].unsqueeze(0) + (
                                1 - new_label_use[i]).unsqueeze(0) * volume_batch_jit22[i]

                label_batch_mixed = label_batch_mixed.long()

                used_mask_mixed = torch.where(label_batch_mixed==2,0,1)
                label_batch_mixed_dc = label_batch_mixed.clone()
                label_batch_mixed_dc[label_batch_mixed_dc==2]=0
                out_1,out_2 = model(volume_batch_mixed),model1(volume_batch_mixed1)
                image_output_soft_1 = torch.softmax(out_1,1)
                image_output_soft_2 = torch.softmax(out_2, 1)

                loss3_ce = ce_reduce_none(image_output_soft_1, label_batch_mixed.long())  #
                loss4_ce = ce_reduce_none(image_output_soft_2, label_batch_mixed.long())  #

                weight_mean = torch.sum(loss_weighted_mixed * used_mask_mixed) / torch.sum(used_mask_mixed)
                loss3_dc = dice_loss_soft0(image_output_soft_1, label_batch_mixed_dc.unsqueeze(1), loss_mask=used_mask_mixed.unsqueeze(1)) * weight_mean
                loss4_dc = dice_loss_soft0(image_output_soft_2, label_batch_mixed_dc.unsqueeze(1), loss_mask=used_mask_mixed.unsqueeze(1)) * weight_mean

                loss3_ce_weighted = 0.5 * torch.sum((loss3_ce) * loss_weighted_mixed * used_mask_mixed) / torch.sum(used_mask_mixed) + loss3_dc * 0.5
                loss4_ce_weighted = 0.5 * torch.sum((loss4_ce) * loss_weighted_mixed * used_mask_mixed) / torch.sum(used_mask_mixed) + loss4_dc * 0.5
                loss_consis_weighted = 0.5 * torch.mean((kl_loss_compute(out_1, out_2, reduce=False) + kl_loss_compute(out_2, out_1, reduce=False)) * loss_weighted_mixed)
                loss += loss3_ce_weighted + loss4_ce_weighted + loss_consis_weighted

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num +=1
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model_search_tau/{}".format(args.datasets + args.exp + args.noise_type)
    os.makedirs(snapshot_path,exist_ok=True)
    train(args, snapshot_path)

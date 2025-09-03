import argparse
import warnings
warnings.filterwarnings('ignore')
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from medpy import metric
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.used_function import assign_superpixel_label_refine, kl_loss_compute
from dataloaders.dataset_npc import BaseDataSets,RandomGenerator_Multi_Rater_our,ZoomGenerator,BaseDataSets_npc_distance_jit_select_one

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='debug', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='0,0.5，0.7')
parser.add_argument('--noise_type', type=str, default='02', help='clean,02,08,DE')
parser.add_argument('--clean_part', type=float, default=0.0, help='part_for_clean')
parser.add_argument('--tr_index', type=int, default=0, help='part_for_clean')
parser.add_argument('--rate_ignore', type=float, default=0.008, help='part_for_clean')
parser.add_argument('--ratecp', type=float, default=0.5)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def train(args, snapshot_path):
    from networks.unet_model import UNet

    model = UNet(3,2).cuda()
    model1 = UNet(3,2).cuda()


    batch_size = args.batch_size
    path = './data/MMIS2024TASK1/'

    db_train = BaseDataSets_npc_distance_jit_select_one(base_dir=path, split="train", transform=RandomGenerator_Multi_Rater_our((128,128)),tr_idx = args.tr_index)
    db_val = BaseDataSets(base_dir=path, split="val", transform=ZoomGenerator((128,128)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn,drop_last=True)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)


    model.train()
    model1.train()

    optimizer = optim.SGD(list(model.parameters())+list(model1.parameters()), lr=0.005, momentum=0.9, weight_decay=1e-5)

    ce_reduce_none = nn.CrossEntropyLoss(ignore_index=2,reduction='none')
    logging.info("{} iterations per epoch".format(len(train_loader)))

    iter_num = 0

    max_epoch = args.max_epochs
    iterator = tqdm(range(max_epoch), ncols=70)
    final_dc0,final_dc1,final_dc2,final_dc3 = [],[],[],[]
    forget_rate =args.rate_ignore
    num_gradual = 20
    from utils.losses import SoftDiceLoss
    dice_loss_soft0 = SoftDiceLoss(smooth=1.)
    rate_schedule = np.ones(100) * forget_rate
    rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** 1, num_gradual)
    co_lambda=0.5

    best_0,best_1,best_2,best_3 ,best_mean= 0,0,0,0,0
    for iter in iterator:
        model.train()
        model1.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            iter_for_use = iter//20
            forget_rate = rate_schedule[iter]
            volume_batch, label_batch,volume_batch_jit,distance_batch,img_SLIC = sampled_batch['image'], sampled_batch['label'],sampled_batch['image_jit'],sampled_batch['label_distance'],sampled_batch['slic']
            volume_batch, volume_batch_jit, label_batch, distance_batch,img_SLIC = \
                volume_batch.cuda().float(), volume_batch_jit.cuda().float(), label_batch.cuda().long(), distance_batch.cuda().long(),distance_batch.cuda().long()

            distance_batch_10 = torch.clip(distance_batch, 1, max(11 - iter_for_use, 1))

            outputs = model(volume_batch)
            outputs1 = model1(volume_batch_jit)
            loss_pick_1 = F.cross_entropy(outputs, label_batch,reduce = False)
            loss_pick_2 = F.cross_entropy(outputs1, label_batch,reduce = False)

            loss_pick = (loss_pick_1 + loss_pick_2 + (1-co_lambda) * kl_loss_compute(outputs, outputs1, reduce=False) + (1-co_lambda )* kl_loss_compute(outputs1, outputs, reduce=False))#.cpu()
            loss_pick2 = torch.flatten(loss_pick[:,])
            distance10_flat = torch.flatten(distance_batch_10[:, ])
            ind_sorted = torch.argsort(loss_pick2.data)
            loss_sorted = loss_pick2[ind_sorted]
            distancewith_loss_sorted = distance10_flat[ind_sorted]#.cpu()

            remember_rate = 1 - forget_rate
            num_remember = int(remember_rate * len(loss_sorted))
            ind_update = (loss_sorted*distancewith_loss_sorted)[:num_remember]
            loss = torch.mean(ind_update)
            # num_remember=int(0.6* len(loss_sorted))
            for_pl_index = torch.zeros_like(ind_sorted)
            for_pl_index[ind_sorted[:num_remember]]=1
            for_pl_mask = for_pl_index.reshape(args.batch_size,128,128)
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

                new_label = masks_generated * (1-used_ori) + used_ori*label_batch
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
                summs = new_label.sum(dim=[1, 2], keepdim=False)
                for i in range(args.batch_size):
                    rate = summs[i] / 16384 # 128*128
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

        if iter >= 10:
            model.eval()
            model1.eval()
            num_test_data = len(val_loader)
            with torch.no_grad():
                test_mdice0,test_mdice1,test_mdice2,test_mdice3 = 0,0,0,0
                for val_step, sample in enumerate(val_loader):
                    patch = sample['image'].cuda()
                    masks = sample['label'].long()[0]
                    maxx = torch.max(masks)
                    preds = []
                    for idx in range(patch.shape[2]):
                        output_slice = model(patch[:,:,idx]) +  model1(patch[:,:,idx])
                        output_pre = torch.argmax(output_slice,1)
                        preds.append(output_pre)
                    preds = torch.cat(preds, 0).to('cpu').numpy()
                    masks = masks.to('cpu').numpy()
                    test_mdice0 += metric.dc(preds, masks[0])
                    test_mdice1 += metric.dc(preds, masks[1])
                    test_mdice2 += metric.dc(preds, masks[2])
                    test_mdice3 += metric.dc(preds, masks[3])

                average_mdice0,average_mdice1,average_mdice2,average_mdice3 = test_mdice0 / num_test_data, test_mdice1 / num_test_data,test_mdice2 / num_test_data,test_mdice3 / num_test_data
                if average_mdice0 > best_0:
                    best_0 = average_mdice0
                if average_mdice1 > best_1:
                    best_1 = average_mdice1
                if average_mdice2 > best_2:
                    best_2 = average_mdice2
                if average_mdice3 > best_3:
                    best_3 = average_mdice3
                average_dice  = (average_mdice0+average_mdice1+average_mdice2+average_mdice3)/4
                if average_dice > best_mean:
                    best_mean = average_dice
                    torch.save(model.state_dict(), snapshot_path + '/' + str(iter) + '.pth')
                    torch.save(model1.state_dict(), snapshot_path + '/' + str(iter) + 'model1.pth')
                f = open(snapshot_path + '/result.txt', "a")
                f.write('epoch' + str(iter) + 'average_mdice0' + str(average_mdice0) + 'average_mdice1' + str(average_mdice1) +
                        'average_mdice2' + str(average_mdice2) + 'average_mdice3' + str(average_mdice3) + '\n')
                f.close()

                final_dc0.append(average_mdice0)
                final_dc1.append(average_mdice1)
                final_dc2.append(average_mdice2)
                final_dc3.append(average_mdice3)

    f = open(snapshot_path + '/result.txt', "a")
    f.write('\n')
    f.write('final   ' + '\n' + 'best_final_dc0    ' + str(np.mean(np.array(final_dc0))) + '\n')
    f.write('final   ' + '\n' + 'best_final_dc1    ' + str(np.mean(np.array(final_dc1))) + '\n')
    f.write('final   ' + '\n' + 'best_final_dc2    ' + str(np.mean(np.array(final_dc2))) + '\n')
    f.write('final   ' + '\n' + 'best_final_dc3    ' + str(np.mean(np.array(final_dc3))) + '\n')
    f.write('best_0,best_1,best_2,best_3'+str(best_0) +'_ '+ str(best_1)+'_ '+ str(best_2)+'_ '+str(best_3))
    f.close()
    print(snapshot_path, 'best_final_dc0    ' + str(np.mean(np.array(final_dc0))) )
    print(snapshot_path, 'best_final_dc1    ' + str(np.mean(np.array(final_dc1))) )
    print(snapshot_path, 'best_final_dc2    ' + str(np.mean(np.array(final_dc2))) )
    print(snapshot_path, 'best_final_dc3    ' + str(np.mean(np.array(final_dc3))) )
    print('best_0,best_1,best_2,best_3',best_0,best_1,best_2,best_3)
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

    snapshot_path = "./model_npc/{}".format(args.exp)
    os.makedirs(snapshot_path,exist_ok=True)
    train(args, snapshot_path)

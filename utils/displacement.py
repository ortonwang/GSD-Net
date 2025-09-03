import numpy as np
import torch
from einops import rearrange
import random

def ABD_R(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch.squeeze(1)[args.labeled_bs:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1)[args.labeled_bs:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # torch.Size([8, 16])
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    for i in range(args.labeled_bs):
        kl_similarities_1 = torch.empty(args.top_num)
        kl_similarities_2 = torch.empty(args.top_num)
        b = torch.argmin(patches_mean_1[i].detach(), dim=0)
        d = torch.argmin(patches_mean_2[i].detach(), dim=0)
        patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
        patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])
        patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
        patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

        for j in range(args.top_num):
            kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
            kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

        a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
        c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)
        a_ori = patches_mean_1_top4_indices[i, a]
        c_ori = patches_mean_2_top4_indices[i, c]

        max_patch_1 = image_patch_2[i][c_ori]  
        image_patch_1[i][b] = max_patch_1  
        max_patch_2 = image_patch_1[i][a_ori]
        image_patch_2[i][d] = max_patch_2 

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size) 
    return image_patch_last

def ABD_R_BCP(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args):
    patches_1 = rearrange(out_max_1, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(out_max_2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(net_input_1.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ',p1=args.patch_size, p2=args.patch_size)  # torch.Size([12, 224, 224])
    image_patch_2 = rearrange(net_input_2.squeeze(1),'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(out_1, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(out_2, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])

    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            kl_similarities_1 = torch.empty(args.top_num)
            kl_similarities_2 = torch.empty(args.top_num)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)
            patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
            patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])

            patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
            patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

            for j in range(args.top_num):
                kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
                kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

            a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

            a_ori = patches_mean_1_top4_indices[i, a]
            c_ori = patches_mean_2_top4_indices[i, c]

            max_patch_1 = image_patch_2[i][c_ori]  
            image_patch_1[i][b] = max_patch_1  
            max_patch_2 = image_patch_1[i][a_ori]
            image_patch_2[i][d] = max_patch_2
        else:
            a = torch.argmax(patches_mean_1[i].detach(), dim=0)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            c = torch.argmax(patches_mean_2[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)

            max_patch_1 = image_patch_2[i][c]  
            image_patch_1[i][b] = max_patch_1  
            max_patch_2 = image_patch_1[i][a]
            image_patch_2[i][d] = max_patch_2
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([24, 224, 224])
    return image_patch_last

def ABD_I(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args):
    # ABD-I Bidirectional Displacement Patch
    patches_supervised_1 = rearrange(outputs1_max[:args.labeled_bs], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2 = rearrange(outputs2_max[:args.labeled_bs], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_1 = rearrange(volume_batch.squeeze(1)[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_supervised_2 = rearrange(volume_batch_strong.squeeze(1)[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_1 = rearrange(label_batch[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_2 = rearrange(label_batch_strong[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_supervised_1 = torch.mean(patches_supervised_1.detach(), dim=2)
    patches_mean_supervised_2 = torch.mean(patches_supervised_2.detach(), dim=2)
    e = torch.argmax(patches_mean_supervised_1.detach(), dim=1) # 获取判断最绝对的区域
    f = torch.argmin(patches_mean_supervised_1.detach(), dim=1) # 获取判断最不绝对的区域
    g = torch.argmax(patches_mean_supervised_2.detach(), dim=1) # 获取判断最绝对的区域
    h = torch.argmin(patches_mean_supervised_2.detach(), dim=1) # 获取判断最不绝对的区域
    for i in range(args.batch_size):
        if random.random() < 0.5:
            min_patch_supervised_1 = image_patch_supervised_2[i][h[i]]    # 提取2里面最不绝对的区域
            image_patch_supervised_1[i][e[i]] = min_patch_supervised_1    # 提取2里面最不绝对的区域，放到1里面
            min_patch_supervised_2 = image_patch_supervised_1[i][f[i]]    # 提取1里面最不绝对的区域
            image_patch_supervised_2[i][g[i]] = min_patch_supervised_2    # 提取1里面最不绝对的区域，放到2里面

            min_label_supervised_1 = label_patch_supervised_2[i][h[i]]
            label_patch_supervised_1[i][e[i]] = min_label_supervised_1
            min_label_supervised_2 = label_patch_supervised_1[i][f[i]]
            label_patch_supervised_2[i][g[i]] = min_label_supervised_2    # 同上，进行标签的处理

    image_patch_supervised = torch.cat([image_patch_supervised_1, image_patch_supervised_2], dim=0)   # 贴完图后的 1， 2
    image_patch_supervised_last = rearrange(image_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    label_patch_supervised = torch.cat([label_patch_supervised_1, label_patch_supervised_2], dim=0)   # 贴完图后的 1， 2 的标签
    label_patch_supervised_last = rearrange(label_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    return image_patch_supervised_last, label_patch_supervised_last


def ABD_label_noise(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args,label_batch):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    # image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # torch.Size([8, 16])
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    # patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4]) # 置信度最大的patch
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状
    # patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1)
    for i in range(args.batch_size):
        kl_similarities_1 = torch.empty(args.top_num)
        kl_similarities_2 = torch.empty(args.top_num)
        # b = torch.argmin(patches_mean_1[i].detach(), dim=0) # 找loss最小的patch的索引
        # d = torch.argmin(patches_mean_2[i].detach(), dim=0) # 找loss最小的patch的索引
        b = torch.argmax(patches_mean_1[i].detach(), dim=0)  # 找loss最大的patch的索引
        d = torch.argmax(patches_mean_2[i].detach(), dim=0)  # 找loss最大的patch的索引
        patches_mean_outputs_max_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
        patches_mean_outputs_max_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])
        patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
        patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])  # 找用于替换的

        for j in range(args.top_num):
            kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_max_2.softmax(dim=-1), reduction='sum')
            kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_max_1.softmax(dim=-1), reduction='sum')
            # loss最小的4个，然后最大的一个，看谁特征最相似
        a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
        c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)  # 获得kl-散度最小的
        a_ori = patches_mean_1_top4_indices[i, a]  # loss最小的用来替换的那一块索引
        c_ori = patches_mean_2_top4_indices[i, c]  # oss最小的用来替换的那一块索引

        # max_patch_1 = image_patch_2[i][c_ori]
        # image_patch_1[i][b] = max_patch_1
        # max_patch_2 = image_patch_1[i][a_ori]
        # image_patch_2[i][d] = max_patch_2    # 进行替换

        max_patch_1 = image_patch_2[i][:,c_ori]
        image_patch_1[i][:,b] = max_patch_1
        max_patch_2 = image_patch_1[i][:,a_ori]
        image_patch_2[i][:,d] = max_patch_2  # 进行替换

        max_patch_gt_1 = label_batch_patches_2[i][ c_ori]
        label_batch_patches_1[i][ b] = max_patch_gt_1
        max_patch_gt_2 = label_batch_patches_1[i][ a_ori]
        label_batch_patches_2[i][ d] = max_patch_gt_2  # 进行替换

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)


    image_patch_last = rearrange(image_patch, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)
    gt_patch_last = rearrange(gt_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last,  gt_patch_last

def ABD_label_noise_2(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args,label_batch):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    # image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # torch.Size([8, 16])
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    # patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4]) # 置信度最大的patch
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状
    # patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1) # 获得loss最小的块状
    for i in range(args.batch_size):
        kl_similarities_1 = torch.empty(args.top_num)
        kl_similarities_2 = torch.empty(args.top_num)
        # b = torch.argmin(patches_mean_1[i].detach(), dim=0) # 找loss最小的patch的索引
        # d = torch.argmin(patches_mean_2[i].detach(), dim=0) # 找loss最小的patch的索引
        b = torch.argmax(patches_mean_1[i].detach(), dim=0)  # 找loss最大的patch的索引
        d = torch.argmax(patches_mean_2[i].detach(), dim=0)  # 找loss最大的patch的索引
        patches_mean_outputs_max_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])  loss最大的一块
        patches_mean_outputs_max_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])  loss最大的一块
        patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
        patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])  # 找用于替换的

        # 先找loss最小的4块，  然后找loss最大的1块，然后4块里面，挑选两个最接近，kl div最小的两块互换
        for j in range(args.top_num):
            kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_max_2.softmax(dim=-1), reduction='sum')
            kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_max_1.softmax(dim=-1), reduction='sum')
            # loss最小的4个，然后最大的一个，看谁特征最相似
        a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
        c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)  # 获得kl-散度最小的
        a_ori = patches_mean_1_top4_indices[i, a]  # loss最小的用来替换的那一块索引
        c_ori = patches_mean_2_top4_indices[i, c]  # oss最小的用来替换的那一块索引

        # max_patch_1 = image_patch_2[i][c_ori]
        # image_patch_1[i][b] = max_patch_1
        # max_patch_2 = image_patch_1[i][a_ori]
        # image_patch_2[i][d] = max_patch_2    # 进行替换

        max_patch_1 = image_patch_2[i][:,c_ori]
        image_patch_1[i][:,b] = max_patch_1
        max_patch_2 = image_patch_1[i][:,a_ori]
        image_patch_2[i][:,d] = max_patch_2  # 进行替换

        max_patch_gt_1 = label_batch_patches_2[i][ c_ori]
        label_batch_patches_1[i][ b] = max_patch_gt_1
        max_patch_gt_2 = label_batch_patches_1[i][ a_ori]
        label_batch_patches_2[i][ d] = max_patch_gt_2  # 进行替换

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)


    image_patch_last = rearrange(image_patch, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)
    gt_patch_last = rearrange(gt_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last,  gt_patch_last

def ABD_label_noise_v15(outputs1_losses, outputs2_losses, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args,label_batch):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    # image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size) # 标签也切块

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # 获取切下来的每一个patch的loss
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    # patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values_big, patches_mean_1_top4_indices_big = torch.topk(patches_mean_1, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状
    patches_mean_2_top4_values_big, patches_mean_2_top4_indices_big = torch.topk(patches_mean_2, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1) # 获得loss最小的块状
    for i in range(args.batch_size):
        for j in range(args.top_num):
            # print('patches_mean_1_top4_indices_big',patches_mean_1_top4_indices_big)
            b = patches_mean_1_top4_indices_big[i,j]  # 找loss最大的patch的索引
            d = patches_mean_2_top4_indices_big[i,j]  # 找loss最大的patch的索引

            a = patches_mean_1_top4_indices[i,j] # loss 最小的
            c = patches_mean_2_top4_indices[i,j] # loss 最小的

            max_patch_1 = image_patch_2[i][:, c]
            image_patch_1[i][:, b] = max_patch_1
            max_patch_2 = image_patch_1[i][:, a]
            image_patch_2[i][:, d] = max_patch_2  # 进行替换

            max_patch_gt_1 = label_batch_patches_2[i][c]
            label_batch_patches_1[i][b] = max_patch_gt_1
            max_patch_gt_2 = label_batch_patches_1[i][a]
            label_batch_patches_2[i][d] = max_patch_gt_2  # 进行替换


    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)


    image_patch_last = rearrange(image_patch, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)
    gt_patch_last = rearrange(gt_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last,  gt_patch_last

def ABD_label_noise_v46_2(outputs1_losses, outputs2_losses, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args,label_batch,regions_to_use):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    regions_to_use_patch_1 = rearrange(regions_to_use, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    regions_to_use_patch_2 = rearrange(regions_to_use, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    # image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size) # 标签也切块

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # 获取切下来的每一个patch的loss
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    # patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values_big, patches_mean_1_top4_indices_big = torch.topk(patches_mean_1, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状
    patches_mean_2_top4_values_big, patches_mean_2_top4_indices_big = torch.topk(patches_mean_2, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1) # 获得loss最小的块状
    for i in range(args.batch_size):
        for j in range(args.top_num):
            # print('patches_mean_1_top4_indices_big',patches_mean_1_top4_indices_big)
            b = patches_mean_1_top4_indices_big[i,j]  # 找loss最大的patch的索引
            d = patches_mean_2_top4_indices_big[i,j]  # 找loss最大的patch的索引

            a = patches_mean_1_top4_indices[i,j] # loss 最小的
            c = patches_mean_2_top4_indices[i,j] # loss 最小的

            max_patch_1 = image_patch_2[i][:, c]
            image_patch_1[i][:, b] = max_patch_1
            max_patch_2 = image_patch_1[i][:, a]
            image_patch_2[i][:, d] = max_patch_2  # 进行替换

            max_patch_gt_1 = label_batch_patches_2[i][c]
            label_batch_patches_1[i][b] = max_patch_gt_1
            max_patch_gt_2 = label_batch_patches_1[i][a]
            label_batch_patches_2[i][d] = max_patch_gt_2  # 进行替换

            max_patch_gt_1 = regions_to_use_patch_2[i][c]
            regions_to_use_patch_1[i][b] = max_patch_gt_1
            max_patch_gt_2 = regions_to_use_patch_1[i][a]
            regions_to_use_patch_2[i][d] = max_patch_gt_2  # 进行替换


    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)
    regions_patch = torch.cat([regions_to_use_patch_1, regions_to_use_patch_2], dim=0)


    image_patch_last = rearrange(image_patch, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)
    gt_patch_last = rearrange(gt_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    regions_last = rearrange(regions_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)

    return image_patch_last,  gt_patch_last,regions_last

def ABD_label_noise_v54(outputs1_losses, outputs2_losses, volume_batch, volume_batch_strong,  args,label_batch,label_batch2):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size) # 标签也切块

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # 获取切下来的每一个patch的loss
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    # patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values_big, patches_mean_1_top4_indices_big = torch.topk(patches_mean_1, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状
    patches_mean_2_top4_values_big, patches_mean_2_top4_indices_big = torch.topk(patches_mean_2, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1) # 获得loss最小的块状
    for i in range(args.batch_size):
        for j in range(args.top_num):
            # print('patches_mean_1_top4_indices_big',patches_mean_1_top4_indices_big)
            b = patches_mean_1_top4_indices_big[i,j]  # 找loss最大的patch的索引
            d = patches_mean_2_top4_indices_big[i,j]  # 找loss最大的patch的索引

            a = patches_mean_1_top4_indices[i,j] # loss 最小的
            c = patches_mean_2_top4_indices[i,j] # loss 最小的

            max_patch_1 = image_patch_2[i][:, c]
            image_patch_1[i][:, b] = max_patch_1
            max_patch_2 = image_patch_1[i][:, a]
            image_patch_2[i][:, d] = max_patch_2  # 进行替换

            max_patch_gt_1 = label_batch_patches_2[i][c]
            label_batch_patches_1[i][b] = max_patch_gt_1
            max_patch_gt_2 = label_batch_patches_1[i][a]
            label_batch_patches_2[i][d] = max_patch_gt_2  # 进行替换


    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)


    image_patch_last = rearrange(image_patch, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)
    gt_patch_last = rearrange(gt_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last,  gt_patch_last


def ABD_label_noise_v48(outputs1_losses, outputs2_losses, volume_batch, volume_batch_strong, args,label_batch,label_batch2):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # 获取切下来的每一个patch的loss
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_mean_1_top4_values_big, patches_mean_1_top4_indices_big = torch.topk(patches_mean_1, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状

    patches_mean_2_top4_values_big, patches_mean_2_top4_indices_big = torch.topk(patches_mean_2, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1) # 获得loss最小的块状
    list_for_replace = np.zeros((args.batch_size,args.top_num,4))
    for i in range(args.batch_size):
        for j in range(args.top_num):
            # print('patches_mean_1_top4_indices_big',patches_mean_1_top4_indices_big)
            b = patches_mean_1_top4_indices_big[i,j]  # 找loss最大的patch的索引
            d = patches_mean_2_top4_indices_big[i,j]  # 找loss最大的patch的索引

            a = patches_mean_1_top4_indices[i,j] # loss 最小的
            c = patches_mean_2_top4_indices[i,j] # loss 最小的
            list_for_replace[i,j] = [a.item(),b.item(),c.item(),d.item()]

            max_patch_1 = image_patch_2[i][:, c]
            image_patch_1[i][:, b] = max_patch_1
            max_patch_2 = image_patch_1[i][:, a]
            image_patch_2[i][:, d] = max_patch_2  # 进行替换

            max_patch_gt_1 = label_batch_patches_2[i][c]
            label_batch_patches_1[i][b] = max_patch_gt_1
            max_patch_gt_2 = label_batch_patches_1[i][a]
            label_batch_patches_2[i][d] = max_patch_gt_2  # 进行替换


    # image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    # gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)


    image_patch_last1 = rearrange(image_patch_1, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)
    gt_patch_last1 = rearrange(label_batch_patches_1, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)

    image_patch_last2 = rearrange(image_patch_2, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    gt_patch_last2 = rearrange(label_batch_patches_2, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last1,  gt_patch_last1,image_patch_last2,gt_patch_last2,list_for_replace


def ABD_label_noise_v34_2(outputs1_losses, outputs2_losses, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args,label_batch,label_batch2):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    # image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size) # 标签也切块

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # 获取切下来的每一个patch的loss
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    # patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values_big, patches_mean_1_top4_indices_big = torch.topk(patches_mean_1, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状
    patches_mean_2_top4_values_big, patches_mean_2_top4_indices_big = torch.topk(patches_mean_2, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1) # 获得loss最小的块状
    for i in range(args.batch_size):
        for j in range(args.top_num):
            # print('patches_mean_1_top4_indices_big',patches_mean_1_top4_indices_big)
            b = patches_mean_1_top4_indices_big[i,j]  # 找loss最大的patch的索引
            d = patches_mean_2_top4_indices_big[i,j]  # 找loss最大的patch的索引

            a = patches_mean_1_top4_indices[i,j] # loss 最小的
            c = patches_mean_2_top4_indices[i,j] # loss 最小的

            max_patch_1 = image_patch_2[i][:, c]
            image_patch_1[i][:, b] = max_patch_1
            max_patch_2 = image_patch_1[i][:, a]
            image_patch_2[i][:, d] = max_patch_2  # 进行替换

            max_patch_gt_1 = label_batch_patches_2[i][c]
            label_batch_patches_1[i][b] = max_patch_gt_1
            max_patch_gt_2 = label_batch_patches_1[i][a]
            label_batch_patches_2[i][d] = max_patch_gt_2  # 进行替换
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)


    image_patch_last = rearrange(image_patch, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)
    gt_patch_last = rearrange(gt_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last,  gt_patch_last

def ABD_label_noise_v37(outputs1_losses, outputs2_losses, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args,label_batch):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    # image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size) # 标签也切块

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # 获取切下来的每一个patch的loss
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    # patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values_big, patches_mean_1_top4_indices_big = torch.topk(patches_mean_1, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状
    patches_mean_2_top4_values_big, patches_mean_2_top4_indices_big = torch.topk(patches_mean_2, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1) # 获得loss最小的块状
    for i in range(args.batch_size):
        gt_here = label_batch_patches_1[i]
        non_zero_region = torch.sum(gt_here==1)
        if non_zero_region < args.patch_size*args.patch_size*4:
            continue
        for j in range(args.top_num):
            # print('patches_mean_1_top4_indices_big',patches_mean_1_top4_indices_big)
            b = patches_mean_1_top4_indices_big[i,j]  # 找loss最大的patch的索引
            d = patches_mean_2_top4_indices_big[i,j]  # 找loss最大的patch的索引

            a = patches_mean_1_top4_indices[i,j] # loss 最小的
            c = patches_mean_2_top4_indices[i,j] # loss 最小的

            max_patch_1 = image_patch_2[i][:, c]
            image_patch_1[i][:, b] = max_patch_1
            max_patch_2 = image_patch_1[i][:, a]
            image_patch_2[i][:, d] = max_patch_2  # 进行替换

            max_patch_gt_1 = label_batch_patches_2[i][c]
            label_batch_patches_1[i][b] = max_patch_gt_1
            max_patch_gt_2 = label_batch_patches_1[i][a]
            label_batch_patches_2[i][d] = max_patch_gt_2  # 进行替换

        # kl_similarities_1 = torch.empty(args.top_num)
        # kl_similarities_2 = torch.empty(args.top_num)
        # # b = torch.argmin(patches_mean_1[i].detach(), dim=0) # 找loss最小的patch的索引
        # # d = torch.argmin(patches_mean_2[i].detach(), dim=0) # 找loss最小的patch的索引
        # b = torch.argmax(patches_mean_1[i].detach(), dim=0)  # 找loss最大的patch的索引
        # d = torch.argmax(patches_mean_2[i].detach(), dim=0)  # 找loss最大的patch的索引
        # patches_mean_outputs_max_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])  loss最大的一块
        # patches_mean_outputs_max_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])  loss最大的一块
        # patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
        # patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])  # 找用于替换的
        #
        # # 先找loss最小的4块，  然后找loss最大的1块，然后4块里面，挑选两个最接近，kl div最小的两块互换
        # for j in range(args.top_num):
        #     kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_max_2.softmax(dim=-1), reduction='sum')
        #     kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_max_1.softmax(dim=-1), reduction='sum')
        #     # loss最小的4个，然后最大的一个，看谁特征最相似
        # a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
        # c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)  # 获得kl-散度最小的
        # a_ori = patches_mean_1_top4_indices[i, a]  # loss最小的用来替换的那一块索引
        # c_ori = patches_mean_2_top4_indices[i, c]  # oss最小的用来替换的那一块索引
        #
        # max_patch_1 = image_patch_2[i][:,c_ori]
        # image_patch_1[i][:,b] = max_patch_1
        # max_patch_2 = image_patch_1[i][:,a_ori]
        # image_patch_2[i][:,d] = max_patch_2  # 进行替换
        #
        # max_patch_gt_1 = label_batch_patches_2[i][ c_ori]
        # label_batch_patches_1[i][ b] = max_patch_gt_1
        # max_patch_gt_2 = label_batch_patches_1[i][ a_ori]
        # label_batch_patches_2[i][ d] = max_patch_gt_2  # 进行替换

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)


    image_patch_last = rearrange(image_patch, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)
    gt_patch_last = rearrange(gt_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last,  gt_patch_last

def ABD_label_noise_v38(outputs1_losses, outputs2_losses, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args,label_batch,label_batch2):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    # image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size) # 标签也切块

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # 获取切下来的每一个patch的loss
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    # patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values_big, patches_mean_1_top4_indices_big = torch.topk(patches_mean_1, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状
    patches_mean_2_top4_values_big, patches_mean_2_top4_indices_big = torch.topk(patches_mean_2, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1) # 获得loss最小的块状
    for i in range(args.batch_size):
        gt_here = label_batch_patches_1[i]
        non_zero_region = torch.sum(gt_here==1)
        if non_zero_region < args.patch_size*args.patch_size*4:
            continue
        for j in range(args.top_num):
            # print('patches_mean_1_top4_indices_big',patches_mean_1_top4_indices_big)
            b = patches_mean_1_top4_indices_big[i,j]  # 找loss最大的patch的索引
            d = patches_mean_2_top4_indices_big[i,j]  # 找loss最大的patch的索引

            a = patches_mean_1_top4_indices[i,j] # loss 最小的
            c = patches_mean_2_top4_indices[i,j] # loss 最小的

            max_patch_1 = image_patch_2[i][:, c]
            image_patch_1[i][:, b] = max_patch_1
            max_patch_2 = image_patch_1[i][:, a]
            image_patch_2[i][:, d] = max_patch_2  # 进行替换

            max_patch_gt_1 = label_batch_patches_2[i][c]
            label_batch_patches_1[i][b] = max_patch_gt_1
            max_patch_gt_2 = label_batch_patches_1[i][a]
            label_batch_patches_2[i][d] = max_patch_gt_2  # 进行替换

        # kl_similarities_1 = torch.empty(args.top_num)
        # kl_similarities_2 = torch.empty(args.top_num)
        # # b = torch.argmin(patches_mean_1[i].detach(), dim=0) # 找loss最小的patch的索引
        # # d = torch.argmin(patches_mean_2[i].detach(), dim=0) # 找loss最小的patch的索引
        # b = torch.argmax(patches_mean_1[i].detach(), dim=0)  # 找loss最大的patch的索引
        # d = torch.argmax(patches_mean_2[i].detach(), dim=0)  # 找loss最大的patch的索引
        # patches_mean_outputs_max_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])  loss最大的一块
        # patches_mean_outputs_max_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])  loss最大的一块
        # patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
        # patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])  # 找用于替换的
        #
        # # 先找loss最小的4块，  然后找loss最大的1块，然后4块里面，挑选两个最接近，kl div最小的两块互换
        # for j in range(args.top_num):
        #     kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_max_2.softmax(dim=-1), reduction='sum')
        #     kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_max_1.softmax(dim=-1), reduction='sum')
        #     # loss最小的4个，然后最大的一个，看谁特征最相似
        # a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
        # c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)  # 获得kl-散度最小的
        # a_ori = patches_mean_1_top4_indices[i, a]  # loss最小的用来替换的那一块索引
        # c_ori = patches_mean_2_top4_indices[i, c]  # oss最小的用来替换的那一块索引
        #
        # max_patch_1 = image_patch_2[i][:,c_ori]
        # image_patch_1[i][:,b] = max_patch_1
        # max_patch_2 = image_patch_1[i][:,a_ori]
        # image_patch_2[i][:,d] = max_patch_2  # 进行替换
        #
        # max_patch_gt_1 = label_batch_patches_2[i][ c_ori]
        # label_batch_patches_1[i][ b] = max_patch_gt_1
        # max_patch_gt_2 = label_batch_patches_1[i][ a_ori]
        # label_batch_patches_2[i][ d] = max_patch_gt_2  # 进行替换

    # image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    # gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)


    image_patch_last1 = rearrange(image_patch_1, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)
    gt_patch_last1 = rearrange(label_batch_patches_1, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)

    image_patch_last2 = rearrange(image_patch_2, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    gt_patch_last2 = rearrange(label_batch_patches_2, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)

    return image_patch_last1,gt_patch_last1 ,  image_patch_last2,gt_patch_last2



def ABD_label_noise_v15_3D(outputs1_losses, outputs2_losses, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args,label_batch):
    # ABD-R Bidirectional Displacement Patch
    # patches_1 = rearrange(outputs1_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_1 = rearrange(outputs1_losses, 'b (h p1) (w p2)(d p3)->b (h w d) (p1 p2 p3)', p1=args.patch_size, p2=args.patch_size,p3=args.patch_size)
    patches_2 = rearrange(outputs2_losses, 'b (h p1) (w p2)(d p3)->b (h w d) (p1 p2 p3)', p1=args.patch_size, p2=args.patch_size, p3=args.patch_size)
    # patches_2 = rearrange(outputs2_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)(d p3)->b (h w d) (p1 p2 p3)', p1=args.patch_size, p2=args.patch_size,p3=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch, 'b (h p1) (w p2)(d p3)->b (h w d) (p1 p2 p3)', p1=args.patch_size, p2=args.patch_size,p3=args.patch_size)
    # label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # label_batch_patches_2 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2)(d p3) -> b c (h w d) (p1 p2 p3) ', p1=args.patch_size, p2=args.patch_size,p3=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2)(d p3) -> b c (h w d) (p1 p2 p3) ', p1=args.patch_size, p2=args.patch_size,p3=args.patch_size)
    # image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    # image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # 获取切下来的每一个patch的loss
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_mean_1_top4_values_big, patches_mean_1_top4_indices_big = torch.topk(patches_mean_1, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状
    patches_mean_2_top4_values_big, patches_mean_2_top4_indices_big = torch.topk(patches_mean_2, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1) # 获得loss最小的块状
    for i in range(args.batch_size):
        for j in range(args.top_num):

            b = patches_mean_1_top4_indices_big[i,j]  # 找loss最大的patch的索引
            d = patches_mean_2_top4_indices_big[i,j]  # 找loss最大的patch的索引

            a = patches_mean_1_top4_indices[i,j] # loss 最小的
            c = patches_mean_2_top4_indices[i,j] # loss 最小的

            max_patch_1 = image_patch_2[i][:, c]
            image_patch_1[i][:, b] = max_patch_1
            max_patch_2 = image_patch_1[i][:, a]
            image_patch_2[i][:, d] = max_patch_2  # 进行替换

            max_patch_gt_1 = label_batch_patches_2[i][c]
            label_batch_patches_1[i][b] = max_patch_gt_1
            max_patch_gt_2 = label_batch_patches_1[i][a]
            label_batch_patches_2[i][d] = max_patch_gt_2  # 进行替换

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)

    image_patch_last = rearrange(image_patch, 'b c (h w d)(p1 p2 p3) -> b  c (h p1) (w p2)(d p3)', c=1,
                                 h=args.h_size, w=args.w_size,d=args.d_size,p1=args.patch_size, p2=args.patch_size,p3=args.patch_size)
    gt_patch_last = rearrange(gt_patch, 'b  (h w d)(p1 p2 p3) -> b  (h p1) (w p2)(d p3)',
                              h=args.h_size, w=args.w_size,d=args.d_size,p1=args.patch_size, p2=args.patch_size,p3=args.patch_size)
    # image_patch_last = rearrange(image_patch, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    # gt_patch_last = rearrange(gt_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last,  gt_patch_last



def ABD_label_noise_v32(outputs1_losses, outputs2_losses, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args,label_batch,label_batch2):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_losses, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)

    label_batch_patches_1 = rearrange(label_batch, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    label_batch_patches_2 = rearrange(label_batch2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # image_patch_1 = rearrange(volume_batch.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    # image_patch_2 = rearrange(volume_batch_strong.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong, 'b c (h p1) (w p2) -> b c (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    # patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size) # 标签也切块

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # 获取切下来的每一个patch的loss
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    # patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    # patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values_big, patches_mean_1_top4_indices_big = torch.topk(patches_mean_1, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_1_top4_values, patches_mean_1_top4_indices = torch.topk(-patches_mean_1, args.top_num, dim=1) # 获得loss最小的块状
    patches_mean_2_top4_values_big, patches_mean_2_top4_indices_big = torch.topk(patches_mean_2, args.top_num, dim=1) # 获得loss最大的块状
    patches_mean_2_top4_values, patches_mean_2_top4_indices = torch.topk(-patches_mean_2, args.top_num, dim=1) # 获得loss最小的块状
    for i in range(args.batch_size):
        for j in range(args.top_num):
            b = patches_mean_1_top4_indices_big[i,j]  # 找loss最大的patch的索引
            d = patches_mean_2_top4_indices_big[i,j]  # 找loss最大的patch的索引

            a = patches_mean_1_top4_indices[i,j] # loss 最小的
            c = patches_mean_2_top4_indices[i,j] # loss 最小的

            max_patch_1 = image_patch_2[i][:, c]
            image_patch_1[i][:, b] = max_patch_1  # 2的loss最小换成1loss最大的
            max_patch_2 = image_patch_1[i][:, a]
            image_patch_2[i][:, d] = max_patch_2  # 1的loss最小换成1loss最大的

            max_patch_gt_1 = label_batch_patches_2[i][c]
            label_batch_patches_1[i][b] = max_patch_gt_1 #  2的loss最小换成1loss最大的
            max_patch_gt_2 = label_batch_patches_1[i][a]
            label_batch_patches_2[i][d] = max_patch_gt_2  # 1的loss最小换成1loss最大的



    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    gt_patch = torch.cat([label_batch_patches_1, label_batch_patches_2], dim=0)


    image_patch_last = rearrange(image_patch, 'b c (h w)(p1 p2) -> b  c (h p1) (w p2)', c=3, h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)
    gt_patch_last = rearrange(gt_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last,  gt_patch_last
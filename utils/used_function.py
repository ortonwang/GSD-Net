import torch
import numpy as np
import torch.nn.functional as F


def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

class PixelWiseDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(PixelWiseDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        if target.ndim == 3:
            target = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        else:
            target = target.float()
        # Intersection and union
        intersection = pred * target
        numerator = 2 * intersection
        denominator = pred + target
        # Sum over channels
        dice_per_pixel = (numerator + self.smooth) / (denominator + self.smooth)
        dice_loss_per_pixel = 1.0 - dice_per_pixel  # (N, C, H, W)

        # Aggregate loss per pixel (average over channels)
        return dice_loss_per_pixel.mean(dim=1)  # (N, H, W)


def assign_superpixel_label_refine(score, segments,threshold=0.5):
    # score
    # segments super pixel的结果
    score_positive = score[1,::]
    score_positive = score_positive.detach().cpu().numpy()
    segments = segments.cpu().numpy()
    unique_segments = np.unique(segments)
    label_map = np.zeros_like(segments, dtype=np.int64)  # superpixel label map
    for seg_id in unique_segments:
        mask_in_seg = (segments == seg_id).astype(np.int64)  # Extract the area segmented by this superpixel
        crossed = mask_in_seg * score_positive
        ratio = np.sum(crossed)/np.sum(mask_in_seg)
        if ratio >= threshold: label=1
        elif ratio <= 1-threshold: label=0
        else: label=2  # param is 0.5, therefore label=2 is not used as ratio range from 0-1
        label_map += mask_in_seg * label
    return  torch.from_numpy(label_map).cuda()
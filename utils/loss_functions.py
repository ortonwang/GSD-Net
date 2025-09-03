import torch
import torch.nn as nn
import torch.nn.functional as F
def get_loss_function(args): #CE,GCE,MAE,RCE,SCE,NGCE,NCE_RCE,NGCE_MAE,NGCE_RCE
    if args.loss_function == 'CE':
        cri = nn.CrossEntropyLoss()
    elif args.loss_function == 'GCE':
        cri = GCELoss()
    elif args.loss_function == 'MAE':
        cri = MAELoss()
    elif args.loss_function == 'RCE':
        cri = RCELoss()
    elif args.loss_function == 'SCE':
        cri = SCELoss()
    elif args.loss_function == 'NGCE':
        cri = NGCELoss()
    elif args.loss_function == 'NCE_RCE':
        cri = NCE_RCE()
    elif args.loss_function == 'NGCE_MAE':
        cri = NGCE_MAE()
    elif args.loss_function == 'NGCE_RCE':
        cri = NGCE_RCE()
    else:
        cri= nn.CrossEntropyLoss()
        print('loss function input is fault')
    # cri = nn.CrossEntropyLoss()
    return cri

class NCE(torch.nn.Module):
    def __init__(self, scale=1.0):
        super(NCE, self).__init__()
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        num_class = pred.size(1)
        label_one_hot = torch.nn.functional.one_hot(labels, num_class).permute(0, 3, 1, 2).float()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()


class NGCELoss(torch.nn.Module):
    def __init__(self, scale=1.0, q=0.7):
        super(NGCELoss, self).__init__()
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, pred.size(1)).permute(0, 3, 1, 2).float()
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = pred.size(1) - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()
class GCELoss(torch.nn.Module):
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, pred.size(1)).permute(0, 3, 1, 2).float()
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, pre, gt):
        pre = pre.float()
        gt_one_hot = torch.nn.functional.one_hot(gt, num_classes=pre.size(1)).permute(0, 3, 1, 2).float()
        loss = torch.mean(torch.abs(pre - gt_one_hot))
        return loss


class RCELoss(torch.nn.Module):
    def __init__(self,  scale=1.0):
        super(RCELoss, self).__init__()
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()


class CrossEntropy:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, logits, target):
        return self.weight * F.cross_entropy(logits, target, reduction='none')

class SCELoss(torch.nn.Module):
    def __init__(self, alpha=1., beta=1.,):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, pred.size(1)).permute(0, 3, 1, 2).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
class SCELoss_ori(torch.nn.Module):
    def __init__(self, alpha=1., beta=1.,):
        super(SCELoss_ori, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction= 'none')

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, pred.size(1)).permute(0, 3, 1, 2).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss
class NCE_RCE:
    def __init__(self, alpha=100.0, beta=1.0):
        self.active_loss = NCE()
        self.passive_loss = RCELoss()

    def __call__(self, logits, target):
        return self.active_loss(logits, target) + self.passive_loss(logits, target)
class NGCE_MAE:
    def __init__(self, alpha=100.0, beta=1.0):
        self.active_loss = NGCELoss()
        self.passive_loss = MAELoss()

    def __call__(self, logits, target):
        return self.active_loss(logits, target) + self.passive_loss(logits, target)
class NGCE_RCE:
    def __init__(self, alpha=100.0, beta=1.0):
        self.active_loss = NGCELoss()
        self.passive_loss = RCELoss()

    def __call__(self, logits, target):
        return self.active_loss(logits, target) + self.passive_loss(logits, target)

if __name__ == "__main__":
    data = torch.rand(2,3,8,8).float()
    gt = torch.randint(0,2,(2,8,8)).long()
    # gt = torch.zeros(4, 64, 64).long()
    cri = SCELoss_ori()
    losss = cri(data,gt)


    print('fds',losss)
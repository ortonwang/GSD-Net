

import torch
import torch.nn.functional as F

class GCELoss(torch.nn.Module):
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, pred.size(1)).permute(0, 4, 1, 2,3).float()
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()

cri = GCELoss()

data = torch.randn(1, 2, 64, 64,64).float()
gt =torch.ones(1,  64, 64,64).long()
loss = cri(data, gt)

print('fds')
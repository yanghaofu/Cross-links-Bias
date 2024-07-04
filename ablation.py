import torch
import torch.nn as nn

class AverageMerger(nn.Module):
    def __init__(self):
        super(AverageMerger, self).__init__()

    def forward(self, emb_ori, emb_aug):
        # 计算两个嵌入的平均值
        return (emb_ori + emb_aug) / 2



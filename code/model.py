"""
模型定义: 骨干 + 年龄特征 + 分类头, Focal Loss, EMA

与 train.py 中的定义一一对应(同源抽取), 供 predict.py 构建网络结构。
"""
import copy

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlaucomaModel(nn.Module):
    """ImageNet 预训练骨干(去掉分类层) + 标准化年龄特征 + 两层 MLP 头"""

    def __init__(self, backbone="efficientnet_b3", pretrained=True,
                 use_age=True, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat = self.backbone.num_features
        self.use_age = use_age
        in_dim = feat + (1 if use_age else 0)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, 1),
        )

    def forward(self, x, age=None):
        f = self.backbone(x)
        if self.use_age and age is not None:
            f = torch.cat([f, age], dim=1)
        return self.head(f).squeeze(1)   # logits


class FocalLoss(nn.Module):
    """专治类别极不均衡; pos_weight 可进一步加权阳性"""

    def __init__(self, alpha=0.5, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class ModelEMA:
    """权重的指数滑动平均: 训练时维护平滑权重, 评估/保存时用, 通常泛化更好"""

    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

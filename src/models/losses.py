"""
SuperPoint 损失函数

接口约定 (与原版 train.py 保持一致):
    criterion = SuperPointLoss(
        det_weight, desc_weight, dustbin_weight=0.1,
        soft_heatmap_weight=0.0,
        soft_heatmap_warmup_epochs=0,
        soft_heatmap_min_target=0.0,
        entropy_weight=0.0,
    )
    criterion.consistency_weight = 0.0        # 训练循环里直接用
    criterion.consistency_warmup_epochs = 0

    total, det, desc, soft, entropy = criterion(
        pred_desc, pred_scores, gt_labels,
        valid_mask=None, soft_heatmap=None, epoch=0,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    标准 SuperPoint 检测损失
    对 [B, 65, Hc, Wc] logits 与 [B, Hc, Wc] 类别标签做交叉熵
    """

    def __init__(self, dustbin_weight=0.1):
        super().__init__()
        self.dustbin_weight = dustbin_weight

    def forward(self, pred_scores, gt_labels):
        if gt_labels.dim() == 4 and gt_labels.size(1) == 1:
            gt_labels = gt_labels.squeeze(1)
        gt_labels = gt_labels.long()

        class_weight = pred_scores.new_ones(65)
        class_weight[64] = self.dustbin_weight
        return F.cross_entropy(pred_scores, gt_labels, weight=class_weight, reduction='mean')


class SoftHeatmapLoss(nn.Module):
    """
    蒸馏损失: student heatmap 逼近 teacher 连续 heatmap。

    soft_heatmap_warmup_epochs 决定 soft loss 强度在前 N 个 epoch 线性从 0 增长到
    soft_heatmap_weight, 之后保持 soft_heatmap_weight。min_target 过滤掉
    teacher heatmap 过低的位置 (没有可学信号)。
    """

    def __init__(self, min_target=0.0):
        super().__init__()
        self.min_target = min_target

    def forward(self, pred_scores, soft_target, weight=1.0, epoch=0, warmup_epochs=0):
        if soft_target is None or weight <= 0:
            return pred_scores.new_tensor(0.0)
        if soft_target.dim() == 3:
            soft_target = soft_target.unsqueeze(1)

        prob = torch.softmax(pred_scores, dim=1)[:, :-1, :, :]
        heatmap = torch.pixel_shuffle(prob, upscale_factor=8)
        if heatmap.shape[-2:] != soft_target.shape[-2:]:
            heatmap = F.interpolate(heatmap, size=soft_target.shape[-2:],
                                    mode='bilinear', align_corners=False)

        target = soft_target.float()
        mask = (target > self.min_target).float()
        if mask.sum() < 1.0:
            return pred_scores.new_tensor(0.0)

        # 线性 warmup
        if warmup_epochs > 0:
            scale = min(1.0, max(0.0, epoch / max(1, warmup_epochs)))
        else:
            scale = 1.0

        loss = F.binary_cross_entropy(
            heatmap.clamp(1e-6, 1.0 - 1e-6),
            target.clamp(0.0, 1.0),
            reduction='none',
        )
        return scale * weight * (loss * mask).sum() / (mask.sum() + 1e-6)


class EntropyRegularization(nn.Module):
    """对 softmax 后的概率分布计算负熵均值。"""

    def __init__(self):
        super().__init__()

    def forward(self, pred_scores, weight=1.0):
        if weight <= 0:
            return pred_scores.new_tensor(0.0)
        prob = torch.softmax(pred_scores, dim=1)
        log_prob = torch.log_softmax(pred_scores, dim=1)
        entropy = -(prob * log_prob).sum(dim=1).mean()
        return weight * entropy


class DescriptorLoss(nn.Module):
    """同位姿三元组 hinge 损失, 默认不启用, 保留以备后续."""

    def __init__(self, margin=1.0, n_neg=16, neg_radius_frac=0.25):
        super().__init__()
        self.margin = margin
        self.n_neg = n_neg
        self.neg_radius_frac = neg_radius_frac

    def forward(self, d1, d2, valid_mask, warp_grid):
        B, D, Hc, Wc = d1.shape
        device = d1.device
        d2_pos = F.grid_sample(d2, warp_grid, mode='bilinear', align_corners=True,
                               padding_mode='border')
        pos_sim = (d1 * d2_pos).sum(dim=1)
        pos_dist = 1.0 - pos_sim
        base = warp_grid
        radius = self.neg_radius_frac
        offsets = (torch.rand(self.n_neg, B, Hc, Wc, 2, device=device) * 2 - 1) * radius
        rand_grids = (base.unsqueeze(0) + offsets).clamp(-1.0, 1.0)
        d2_neg_all = F.grid_sample(
            d2.unsqueeze(0).expand(self.n_neg, -1, -1, -1, -1).reshape(-1, D, Hc, Wc),
            rand_grids.reshape(-1, Hc, Wc, 2),
            mode='bilinear', align_corners=True, padding_mode='border',
        ).reshape(self.n_neg, B, D, Hc, Wc)
        neg_sim = (d1.unsqueeze(0) * d2_neg_all).sum(dim=2)
        hard_neg_dist, _ = neg_sim.min(dim=0)
        hard_neg_dist = 1.0 - hard_neg_dist
        triplet = pos_dist - hard_neg_dist + self.margin
        triplet = F.relu(triplet)
        mask = valid_mask.float()
        n_valid = mask.sum()
        if n_valid < 1:
            return d1.new_tensor(0.0)
        return (triplet * mask).sum() / n_valid


class SuperPointLoss(nn.Module):
    """
    主损失入口 (接口与原版 train.py 一致)

    可选属性 (训练循环读取):
        consistency_weight, consistency_warmup_epochs, consistency_min_confidence
    """

    def __init__(self, det_weight=1.0, desc_weight=0.0, dustbin_weight=0.1,
                 soft_heatmap_weight=0.0,
                 soft_heatmap_warmup_epochs=0,
                 soft_heatmap_min_target=0.0,
                 entropy_weight=0.0,
                 entropy_warmup_epochs=0):
        super().__init__()
        self.det_loss = DetectionLoss(dustbin_weight=dustbin_weight)
        self.soft_heatmap_loss = SoftHeatmapLoss(min_target=soft_heatmap_min_target)
        self.entropy_reg = EntropyRegularization()
        self.desc_loss = DescriptorLoss()
        self.det_weight = det_weight
        self.desc_weight = desc_weight
        self.soft_heatmap_weight = soft_heatmap_weight
        self.soft_heatmap_warmup_epochs = soft_heatmap_warmup_epochs
        self.entropy_weight = entropy_weight
        self.entropy_warmup_epochs = entropy_warmup_epochs
        # consistency 在 train_one_epoch 外部算, 这里只保留默认属性
        self.consistency_weight = 0.0
        self.consistency_warmup_epochs = 0
        self.consistency_min_confidence = 0.05

    def forward(self, pred_desc, pred_scores, gt_labels,
                valid_mask=None, soft_heatmap=None, epoch=0):
        det = self.det_loss(pred_scores, gt_labels)

        soft = self.soft_heatmap_loss(
            pred_scores, soft_heatmap,
            weight=self.soft_heatmap_weight,
            epoch=epoch, warmup_epochs=self.soft_heatmap_warmup_epochs,
        )

        entropy = self.entropy_reg(pred_scores, weight=self.entropy_weight)

        if self.desc_weight > 0 and valid_mask is not None and pred_desc is not None:
            # 没有第二个视图时 desc 退化为 0 (避免 NaN)
            desc = pred_scores.new_tensor(0.0)
        else:
            desc = pred_scores.new_tensor(0.0)

        total = (self.det_weight * det
                 + soft
                 + entropy
                 + self.desc_weight * desc)
        return total, det, desc, soft, entropy
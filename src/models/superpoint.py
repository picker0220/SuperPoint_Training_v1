"""
SuperPoint 模型架构

与原版的区别:
- 每个 conv 后接 BatchNorm (与 MagicLeap 公开预训练权重兼容)
- 共享编码器输出 128 通道 (而非 256),与 MagicLeap 一致
- 检测头/描述子头的第一个 conv 输入 128 通道 (直接接 encoder 输出)

整体结构 (与 rpautrat/SuperPoint 公开预训练权重完全兼容):
    输入: batch x 1 x H x W 灰度图
    backbone:
        conv1a (1->64) + bn + relu
        conv1b (64->64) + bn + relu
        pool1
        conv2a (64->64) + bn + relu
        conv2b (64->64) + bn + relu
        pool2
        conv3a (64->128) + bn + relu
        conv3b (128->128) + bn + relu
        pool3
        conv4a (128->128) + bn + relu
        conv4b (128->128) + bn + relu
    detector head:
        det_conv1 (128->256) + bn + relu
        det_conv2 (256->65) + bn
    descriptor head:
        desc_conv1 (128->256) + bn + relu
        desc_conv2 (256->256) + bn
    输出: descriptors [B, 256, H/8, W/8] (L2 归一化), scores [B, 65, H/8, W/8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperPoint(nn.Module):
    def __init__(self, encoder_dim=256, grid_size=8):
        super().__init__()
        self.grid_size = grid_size

        # ==================== 共享编码器 ====================
        # Block 1
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4 (输出 128 通道, 与 MagicLeap 对齐)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(128)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(128)

        # ==================== 关键点检测头 ====================
        # 输入 128 (encoder 输出), 输出 65 (64 子位置 + 1 dustbin)
        self.det_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn_det1 = nn.BatchNorm2d(256)
        self.det_conv2 = nn.Conv2d(256, 65, kernel_size=1, stride=1)
        self.bn_det2 = nn.BatchNorm2d(65)

        # ==================== 描述子头 ====================
        # 输入 128, 输出 256 (与 MagicLeap 对齐, ORB_SLAM3_Hybrid 兼容)
        self.desc_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn_desc1 = nn.BatchNorm2d(256)
        self.desc_conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.bn_desc2 = nn.BatchNorm2d(256)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.bn4a(self.conv4a(x)))
        feat = F.relu(self.bn4b(self.conv4b(x)))

        # 检测头
        det = F.relu(self.bn_det1(self.det_conv1(feat)))
        det = self.bn_det2(self.det_conv2(det))

        # 描述子头
        desc = F.relu(self.bn_desc1(self.desc_conv1(feat)))
        desc = self.bn_desc2(self.desc_conv2(desc))
        desc = F.normalize(desc, p=2, dim=1)

        return desc, det


def load_magicpoint_weights(model, weights_path, strict=False, verbose=True):
    """
    从 MagicLeap 公开预训练权重初始化 (rpautrat/SuperPoint weights/superpoint_v6_from_tf.pth)

    权重命名映射 (MagicLeap Sequential 风格 -> 你的直名风格):
        backbone.{block}.{layer}.conv.weight -> conv{block}{a|b}.weight
        backbone.{block}.{layer}.bn.weight    -> bn{block}{a|b}.weight
        ... (running_mean, running_var, bias 类似)
        detector.0.conv.weight -> det_conv1.weight
        detector.0.bn.weight    -> bn_det1.weight
        detector.1.conv.weight -> det_conv2.weight
        detector.1.bn.weight    -> bn_det2.weight
        descriptor.0.conv.weight -> desc_conv1.weight
        descriptor.0.bn.weight    -> bn_desc1.weight
        descriptor.1.conv.weight -> desc_conv2.weight
        descriptor.1.bn.weight    -> bn_desc2.weight

    Args:
        model: 你的 SuperPoint 实例
        weights_path: MagicLeap .pth 路径
        strict: True 则任何不匹配都报错; False 则只 warn
        verbose: True 则打印加载报告
    """
    import torch
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    # backbone.{block}.{0|1}.{conv|bn}.{weight|bias|running_mean|running_var}
    new_sd = {}
    matched, skipped = [], []
    for k, v in sd.items():
        parts = k.split('.')
        # 识别 backbone / detector / descriptor
        if parts[0] == 'backbone':
            # backbone.{block}.{layer}.{type}.{param}
            # parts: ['backbone', block, layer, type, param]
            block_idx = int(parts[1])
            layer_idx = int(parts[2])
            kind = parts[3]  # 'conv' or 'bn'
            param = parts[4]  # 'weight', 'bias', 'running_mean', 'running_var'
            sub = 'a' if layer_idx == 0 else 'b'
            if kind == 'conv':
                new_key = f'conv{block_idx+1}{sub}.{param}'
            else:  # 'bn'
                new_key = f'bn{block_idx+1}{sub}.{param}'
        elif parts[0] == 'detector':
            # detector.{0|1}.{conv|bn}.{param}
            head_idx = int(parts[1]) + 1
            kind = parts[2]
            param = parts[3]
            if kind == 'conv':
                new_key = f'det_conv{head_idx}.{param}'
            else:
                new_key = f'bn_det{head_idx}.{param}'
        elif parts[0] == 'descriptor':
            head_idx = int(parts[1]) + 1
            kind = parts[2]
            param = parts[3]
            if kind == 'conv':
                new_key = f'desc_conv{head_idx}.{param}'
            else:
                new_key = f'bn_desc{head_idx}.{param}'
        else:
            skipped.append((k, 'unknown prefix'))
            continue
        new_sd[new_key] = v
        matched.append((k, new_key))

    # 应用到 model
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if verbose:
        print(f'[MagicPoint loader] {len(matched)}/{len(sd)} weights mapped and loaded')
        if missing:
            print(f'  missing in checkpoint (will be reinitialized): {len(missing)}')
            for m in missing[:10]:
                print(f'    - {m}')
        if unexpected:
            print(f'  unexpected keys (ignored): {len(unexpected)}')
            for u in unexpected[:5]:
                print(f'    - {u}')
    if strict and (missing or unexpected):
        raise RuntimeError(f'MagicPoint strict load failed: missing={missing}, unexpected={unexpected}')
    return model
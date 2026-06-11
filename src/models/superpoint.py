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

    def forward_to_feat(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        return x

    def forward_onnx(self, x):
        # ONNX 导出专用: (prob [B,1,H,W], desc [B,256,H/8,W/8])
        feat = self.forward_to_feat(x)
        det = F.relu(self.bn_det1(self.det_conv1(feat)))
        det_logits = self.bn_det2(self.det_conv2(det))
        prob = torch.softmax(det_logits, dim=1)[:, :-1, :, :]
        prob = torch.pixel_shuffle(prob, 8)  # [B, 1, H, W]
        desc = F.relu(self.bn_desc1(self.desc_conv1(feat)))
        desc = self.bn_desc2(self.desc_conv2(desc))
        # C++ postprocessOutput 在 1/8 分辨率 spatial lookup, 保持 [B, 256, H/8, W/8]
        desc = F.normalize(desc, p=2, dim=1)
        return prob, desc



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

def load_legacy_weights(model, checkpoint_path, verbose=True):
    """
    从你**老架构**(conv1a-conv4b 是 plain Conv2d 无 BN)训出来的 .pth 加载到当前新架构。
    典型场景:你之前训的 superpoint_epoch_35.pth,encoder 输出 256 通道。
    当前新架构 encoder 输出 128 通道,det/desc head 输入 128(兼容 MagicLeap)。

    截断法(Truncation)映射:
        老 conv1a/1b/2a/2b/3a/3b   -> 新同结构(直接 copy)
        老 conv4a [256,128,3,3]   -> 新 conv4a [128,128,3,3]: 取前 128 个输出通道
        老 conv4b [256,256,3,3]   -> 新 conv4b [128,128,3,3]: 输入/输出各取前 128
        老 det_conv1 [256,256,3,3] -> 新 det_conv1 [256,128,3,3]: 输入取前 128
        老 desc_conv1 [256,256,3,3]-> 新 desc_conv1 [256,128,3,3]: 输入取前 128
        老 det_conv2 / desc_conv2 -> 同结构(直接 copy)
        老 BN 参数(若存在)        -> 按通道截断

    BN 行为:老架构可能没 BN,新架构有 BN(默认 init running stats=0/1,
    在 forward 时接近 identity,会重新估计)。建议 fine-tune 前
    调用 calibrate_bn() 跑几批数据估 BN stats,精度更高。

    Returns:
        model: 加载后的 model
        legacy_loaded: int,成功映射的层数
    """
    import torch
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

    def take(t, n, dim=0):
        # take first n along given dim
        if t.ndim > dim and t.shape[dim] >= n:
            return t.narrow(dim, 0, n).clone()
        return t
        return t

    new_sd = {}
    legacy_loaded = 0
    for k, v in sd.items():
        if k == 'conv1a.weight' or k == 'conv1a.bias':
            new_sd[k] = v; legacy_loaded += 1
        elif k == 'conv1b.weight' or k == 'conv1b.bias':
            new_sd[k] = v; legacy_loaded += 1
        elif k == 'conv2a.weight' or k == 'conv2a.bias':
            new_sd[k] = v; legacy_loaded += 1
        elif k == 'conv2b.weight' or k == 'conv2b.bias':
            new_sd[k] = v; legacy_loaded += 1
        elif k == 'conv3a.weight' or k == 'conv3a.bias':
            new_sd[k] = v; legacy_loaded += 1
        elif k == 'conv3b.weight' or k == 'conv3b.bias':
            new_sd[k] = v; legacy_loaded += 1
        # 关键截断:老 conv4a 输出 256,新 conv4a 输出 128,取前 128
        elif k == 'conv4a.weight':
            new_sd[k] = take(v, 128); legacy_loaded += 1
        elif k == 'conv4a.bias':
            new_sd[k] = take(v, 128); legacy_loaded += 1
        # 老 conv4b 输入 256 输出 256,新输入 128 输出 128,各取前 128
        elif k == 'conv4b.weight':
            # weight 形状 [out, in, k, k] -> [128, 128, 3, 3]
            new_sd[k] = take(take(v, 128, dim=0), 128, dim=1); legacy_loaded += 1
        elif k == 'conv4b.bias':
            new_sd[k] = take(v, 128); legacy_loaded += 1
        # BN(若老架构有):截前 128
        elif k in ('bn4a.weight', 'bn4a.bias', 'bn4a.running_mean', 'bn4a.running_var'):
            new_sd[k] = take(v, 128); legacy_loaded += 1
        elif k in ('bn4b.weight', 'bn4b.bias', 'bn4b.running_mean', 'bn4b.running_var'):
            new_sd[k] = take(v, 128); legacy_loaded += 1
        # det_conv1 输入 256 -> 128(输出 256 保留)
        elif k == 'det_conv1.weight':
            new_sd[k] = take(v, 128, dim=1); legacy_loaded += 1
        elif k == 'det_conv1.bias':
            new_sd[k] = v; legacy_loaded += 1
        # BN_det1 若存在
        elif k in ('bn_det1.weight', 'bn_det1.bias', 'bn_det1.running_mean', 'bn_det1.running_var'):
            new_sd[k] = v; legacy_loaded += 1
        # det_conv2 同结构
        elif k in ('det_conv2.weight', 'det_conv2.bias',
                   'bn_det2.weight', 'bn_det2.bias', 'bn_det2.running_mean', 'bn_det2.running_var'):
            new_sd[k] = v; legacy_loaded += 1
        # desc_conv1 输入 256 -> 128
        elif k == 'desc_conv1.weight':
            new_sd[k] = take(v, 128, dim=1); legacy_loaded += 1
        elif k == 'desc_conv1.bias':
            new_sd[k] = v; legacy_loaded += 1
        elif k in ('bn_desc1.weight', 'bn_desc1.bias', 'bn_desc1.running_mean', 'bn_desc1.running_var'):
            new_sd[k] = v; legacy_loaded += 1
        # desc_conv2 同结构
        elif k in ('desc_conv2.weight', 'desc_conv2.bias',
                   'bn_desc2.weight', 'bn_desc2.bias', 'bn_desc2.running_mean', 'bn_desc2.running_var'):
            new_sd[k] = v; legacy_loaded += 1
        # 其它 key 全部忽略

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if verbose:
        print(f'[Legacy loader] {legacy_loaded} weights mapped and loaded')
        if missing:
            print(f'  missing in new model (BN params 等,init 默认): {len(missing)}')
            for m in missing[:8]:
                print(f'    - {m}')
        if unexpected:
            print(f'  unexpected keys (ignored): {len(unexpected)}')
            for u in unexpected[:5]:
                print(f'    - {u}')
        print(f'  -> BN running stats 默认 init=0/1,建议 fine-tune 前调 calibrate_bn()')
    return model, legacy_loaded



def calibrate_bn(model, image_paths, device='cpu', input_height=480, input_width=640, num_batches=8):  # NOTE: no @torch.no_grad() decorator, BN running stats must update
    """
    在真实数据上跑几批 forward,校准 BN running_mean / running_var。
    legacy 加载的 BN 默认 init=0/1(实际近似 identity),校准后精度更高。

    用法:
        calibrate_bn(model, sorted(glob('~/superpoint/dataset/dark/*.JPG'))[:50])
    """
    import cv2
    import numpy as np
    from visualize import preprocess_image
    from torch.utils.data import DataLoader, TensorDataset

    model.train()  # 关键:BN 收集 running stats
    # 收集图
    imgs = []
    for p in image_paths[:num_batches * 4]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (input_width, input_height))
        t = preprocess_image(img)
        imgs.append(t.squeeze(0))  # [1, H, W] for stack
        if len(imgs) >= num_batches * 4:
            break
    if not imgs:
        print('[calibrate_bn] no valid images')
        model.eval()
        return
    batch = torch.stack(imgs).to(device)
    print(f'[calibrate_bn] running {num_batches} batches of {batch.shape[0]} images')
    # 多次 forward 让 running stats 收敛
    # 多次 forward 让 running stats 收敛 (注:不能用 @torch.no_grad() 装饰器,否则 BN 不会更新 running_mean/var)
    with torch.no_grad():
        for _ in range(3):
            for i in range(0, batch.shape[0], max(1, batch.shape[0] // num_batches)):
                model(batch[i:i+num_batches])


class SuperPointLegacy(nn.Module):
    """
    你原来 epoch_35 训的 SuperPoint 架构 (老架构):
    - 无 BN (老架构没 BN 层)
    - 8 conv + 3 maxpool,无 normalization
    - conv4a 输出 256 通道
    - det_conv1 / desc_conv1 输入 256
    - encoder 末端输出 256 通道

    输出与新 SuperPoint 完全兼容 (65 类 logits + 256-dim 描述子),
    训练 / HA / eval 流水线不用改,只是模型类不同。

    用法:
        from src.models.superpoint import SuperPointLegacy
        model = SuperPointLegacy()
        sd = torch.load('checkpoints/superpoint_epoch_35.pth')['model_state_dict']
        model.load_state_dict(sd)  # 直接 load,不需要 shim
    """
    def __init__(self, encoder_dim=256, grid_size=8):
        super().__init__()
        self.grid_size = grid_size

        # Block 1
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4 - 输出 256 (跟新架构 128 不同)
        self.conv4a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # 检测头 - 输入 256 (跟新架构 128 不同)
        self.det_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.det_conv2 = nn.Conv2d(256, 65, kernel_size=1, stride=1)

        # 描述子头 - 输入 256 (跟新架构 128 不同)
        self.desc_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.desc_conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        # Block 2
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        # Block 3
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)
        # Block 4
        x = F.relu(self.conv4a(x))
        feat = F.relu(self.conv4b(x))

        # 检测头
        det = F.relu(self.det_conv1(feat))
        det = self.det_conv2(det)
        # 描述子头
        desc = F.relu(self.desc_conv1(feat))
        desc = self.desc_conv2(desc)
        desc = F.normalize(desc, p=2, dim=1)

        return desc, det

    def forward_onnx(self, x):
        """ONNX 导出专用,与新 SuperPoint 的 forward_onnx 输出格式完全一致"""
        feat = self.forward_to_feat(x)
        det = F.relu(self.det_conv1(feat))
        det_logits = self.det_conv2(det)
        prob = torch.softmax(det_logits, dim=1)[:, :-1, :, :]
        prob = torch.pixel_shuffle(prob, 8)
        desc = F.relu(self.desc_conv1(feat))
        desc = self.desc_conv2(desc)
        # C++ postprocessOutput 在 1/8 分辨率 spatial lookup,不 pixel_shuffle desc (256/8^2=4 通道错)
        desc = F.normalize(desc, p=2, dim=1)  # 保持 [B, 256, H/8, W/8] 输出
        desc = F.normalize(desc, p=2, dim=1)
        return prob, desc

    def forward_to_feat(self, x):
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        return x


def load_legacy_checkpoint(ckpt_path, device='cpu', verbose=True):
    """
    加载 epoch_35 (老架构) .pth 到一个 SuperPointLegacy 实例。
    不截断、不 shim,1:1 完美加载,精度跟原 epoch_35 一样。

    Args:
        ckpt_path: 你的 superpoint_epoch_35.pth 路径
        device: 加载到的设备
        verbose: 打印加载报告

    Returns:
        model: SuperPointLegacy 实例,eval 模式,weights 已加载
    """
    import torch
    model = SuperPointLegacy(encoder_dim=256, grid_size=8)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        n_loaded = len(sd) - len(unexpected)
        print(f'[load_legacy_checkpoint] {n_loaded}/{len(sd)} weights loaded into SuperPointLegacy')
        if missing:
            print(f'  missing in checkpoint: {len(missing)}')
            for m in missing[:5]:
                print(f'    - {m}')
        if unexpected:
            print(f'  unexpected keys: {len(unexpected)}')
            for u in unexpected[:5]:
                print(f'    - {u}')

    model = model.to(device).eval()
    return model


def detect_arch_from_ckpt(ckpt_path):
    """
    检测 .pth 是新架构 (有 BN, 见 bn1a.weight) 还是老架构 (无 BN)。
    返回 'new' / 'legacy' / 'unknown'
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

    has_bn = any(k.startswith('bn') for k in sd.keys())
    has_backbone = any(k.startswith('backbone.') for k in sd.keys())
    has_detector = any(k.startswith('detector.') for k in sd.keys())
    has_descriptor = any(k.startswith('descriptor.') for k in sd.keys())
    has_magicpoint_names = has_backbone or has_detector or has_descriptor

    if has_magicpoint_names:
        return 'magicpoint'
    if has_bn:
        return 'new'
    if 'conv1a.weight' in sd:
        return 'legacy'
    return 'unknown'

def load_superpoint(ckpt_path, device='cpu', verbose=False):
    """
    通用 SuperPoint 加载器:自动识别老 / 新 / MagicLeap 架构。

    识别逻辑来自 detect_arch_from_ckpt():
        legacy     -> SuperPointLegacy + load_legacy_checkpoint (1:1 完美加载, 无截断)
        magicpoint -> SuperPoint + load_magicpoint_weights (命名映射; 通道不匹配 strict=False 跳过)
        new/unknown-> SuperPoint + 原始 state_dict (strict=False 容忍 missing/unexpected)

    用法:
        model = load_superpoint('checkpoints/superpoint_epoch_35.pth', device='cuda')
    """
    arch = detect_arch_from_ckpt(ckpt_path)
    if verbose:
        print('[load_superpoint] detected arch:', arch)

    if arch == 'legacy':
        return load_legacy_checkpoint(ckpt_path, device=device, verbose=verbose)

    if arch == 'magicpoint':
        model = SuperPoint(encoder_dim=256, grid_size=8)
        model = load_magicpoint_weights(model, ckpt_path, strict=False, verbose=verbose)
        return model.to(device).eval()

    # 'new' or 'unknown'
    model = SuperPoint(encoder_dim=256, grid_size=8)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        print('[load_superpoint] new arch: missing=%d unexpected=%d' % (len(missing), len(unexpected)))
    return model.to(device).eval()

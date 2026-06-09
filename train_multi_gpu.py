"""
SuperPoint 双卡4090训练脚本

优化配置:
- 双卡并行训练 (DataParallel)
- 更大的批次大小
- 混合精度训练 (AMP)
- 多worker数据加载

如需只使用单卡训练，将 USE_MULTI_GPU = False 即可
"""

# ============================================================
# 单卡训练配置 - 设为 False 则只使用单张GPU
# ============================================================
USE_MULTI_GPU = True   # True: 使用双卡, False: 只使用单卡

# ============================================================
# 训练超参数 - 可根据显存调整
# ============================================================
BATCH_SIZE_PER_GPU = 16      # 每张卡的批次大小 (4090 24GB显存建议16-24)
LEARNING_RATE = 0.001        # 学习率
NUM_EPOCHS = 200             # 训练轮数
GRAD_ACCUM_STEPS = 1        # 梯度累积步数 (总batch = 16 * 2 * 1 = 32)
USE_AMP = True               # 使用混合精度训练，省显存且加速

import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config
from src.models.superpoint import SuperPoint
from src.models.losses import SuperPointLoss
from src.data.synthetic_dataset import SyntheticDataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='双卡4090训练SuperPoint')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_PER_GPU * (2 if USE_MULTI_GPU else 1), help='总批次大小')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    return parser.parse_args()


def get_device_info():
    """获取GPU设备信息"""
    if not torch.cuda.is_available():
        return "CPU", 0

    n_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return gpu_name, n_gpus, gpu_mem


def create_model_multi_gpu(config):
    """
    创建支持多GPU的SuperPoint模型

    使用 DataParallel 自动在多张GPU上分发数据
    """
    print("=" * 60)
    print("创建SuperPoint模型 (多GPU模式)")
    print("=" * 60)
    print(f"  - 编码器维度: {config.ENCODER_DIM}")
    print(f"  - 下采样倍率: {config.GRID_SIZE}")
    print(f"  - 描述子维度: 128")

    model = SuperPoint(
        encoder_dim=config.ENCODER_DIM,
        grid_size=config.GRID_SIZE
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 总参数量: {total_params:,}")

    # 多GPU并行 - 简单方案
    # 如需更高效的分布式训练，可改用 DistributedDataParallel (DDP)
    model = DataParallel(model)

    print("  - 使用 DataParallel 多卡训练")
    print()

    return model


def create_model_single_gpu(config):
    """创建单GPU模型"""
    print("=" * 60)
    print("创建SuperPoint模型 (单GPU模式)")
    print("=" * 60)

    model = SuperPoint(
        encoder_dim=config.ENCODER_DIM,
        grid_size=config.GRID_SIZE
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 总参数量: {total_params:,}")
    print()

    return model


def create_data_loader(config, batch_size):
    """创建数据加载器"""
    print("=" * 60)
    print("创建数据集")
    print("=" * 60)
    print(f"  - 数据类型: 合成图像 (网格关键点)")
    print(f"  - 图像尺寸: {config.IMAGE_HEIGHT} x {config.IMAGE_WIDTH}")
    print(f"  - 训练样本数: 1000")
    print(f"  - 批次大小: {batch_size}")

    train_dataset = SyntheticDataset(
        num_samples=1000,
        image_height=config.IMAGE_HEIGHT,
        image_width=config.IMAGE_WIDTH,
        grid_size=config.GRID_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 多进程数据加载
        pin_memory=True  # 加速GPU数据传输
    )

    print(f"  - 训练批次数: {len(train_loader)}")
    print()

    return train_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,
                    scaler=None, print_interval=10, use_amp=False):
    """训练一个epoch"""
    model.train()

    running_loss = 0.0
    running_det_loss = 0.0
    running_desc_loss = 0.0
    batch_count = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (images, heatmaps, keypoints) in enumerate(pbar):
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        if use_amp and scaler is not None:
            with autocast():
                descriptors, scores = model(images)
                total_loss, det_loss, desc_loss = criterion(
                    descriptors, scores, heatmaps, valid_mask=None
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            descriptors, scores = model(images)
            total_loss, det_loss, desc_loss = criterion(
                descriptors, scores, heatmaps, valid_mask=None
            )
            total_loss.backward()
            optimizer.step()

        running_loss += total_loss.item()
        running_det_loss += det_loss.item()
        running_desc_loss += desc_loss.item()
        batch_count += 1

        if batch_idx % print_interval == 0:
            pbar.set_postfix({
                'loss': f'{running_loss / batch_count:.4f}',
                'det': f'{running_det_loss / batch_count:.4f}',
                'desc': f'{running_desc_loss / batch_count:.4f}'
            })

    return running_loss / batch_count, running_det_loss / batch_count, running_desc_loss / batch_count


def save_checkpoint(model, optimizer, epoch, loss, config, filename):
    """保存检查点"""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # 如果是DataParallel模型，只保存module的state_dict
    model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(checkpoint, filepath)
    print(f"  保存检查点: {filepath}")


def main():
    """主训练函数"""
    args = parse_args()

    config = Config()
    config.CHECKPOINT_DIR = args.checkpoint_dir

    # ============================================================
    # 设备设置
    # ============================================================
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 设备才能进行训练")
        return

    n_gpus = torch.cuda.device_count()

    # 根据 USE_MULTI_GPU 决定使用单卡还是双卡
    if USE_MULTI_GPU and n_gpus >= 2:
        actual_gpus = 2
        print(f"=" * 60)
        print(f"多GPU训练模式")
        print(f"=" * 60)
        print(f"  检测到 {n_gpus} 张 GPU")
        print(f"  将使用 {actual_gpus} 张 GPU 并行训练")
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 1: {torch.cuda.get_device_name(1)}")
    elif USE_MULTI_GPU and n_gpus == 1:
        actual_gpus = 1
        print(f"=" * 60)
        print(f"单GPU模式 (检测到仅1张GPU)")
        print(f"=" * 60)
        print(f"  使用 GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        actual_gpus = 1
        print(f"=" * 60)
        print(f"单GPU训练模式")
        print(f"=" * 60)

    device = torch.device('cuda')
    print()

    # ============================================================
    # 创建模型
    # ============================================================
    if actual_gpus > 1:
        model = create_model_multi_gpu(config)
    else:
        model = create_model_single_gpu(config)
    model = model.to(device)

    # ============================================================
    # 创建数据加载器
    # ============================================================
    # 批次大小 = 每GPU批次 * GPU数量
    total_batch_size = args.batch_size
    if actual_gpus > 1 and USE_MULTI_GPU:
        # 自动调整批次大小
        total_batch_size = BATCH_SIZE_PER_GPU * actual_gpus

    train_loader = create_data_loader(config, total_batch_size)

    # ============================================================
    # 损失函数和优化器
    # ============================================================
    criterion = SuperPointLoss(
        det_weight=config.DET_LOSS_WEIGHT,
        desc_weight=config.DESC_LOSS_WEIGHT
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 混合精度训练的Scaler
    scaler = GradScaler() if USE_AMP else None

    # ============================================================
    # 开始训练
    # ============================================================
    print("=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"  - Epochs: {args.epochs}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 总批次大小: {total_batch_size}")
    print(f"  - 每GPU批次: {total_batch_size // actual_gpus if actual_gpus > 1 else total_batch_size}")
    print(f"  - 混合精度: {'开启' if USE_AMP else '关闭'}")
    print(f"  - 保存间隔: {config.SAVE_INTERVAL}")
    print()

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, det_loss, desc_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, use_amp=USE_AMP
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Loss: {train_loss:.4f} (Det: {det_loss:.4f}, Desc: {desc_loss:.4f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {epoch_time:.1f}s")

        if epoch % config.SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, config,
                f'superpoint_dual4090_epoch_{epoch}.pth'
            )

    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("训练完成!")
    print(f"总训练时间: {total_time / 60:.2f} 分钟")
    print("=" * 60)

    save_checkpoint(
        model, optimizer, args.epochs, train_loss, config,
        'superpoint_dual4090_final.pth'
    )


if __name__ == '__main__':
    main()
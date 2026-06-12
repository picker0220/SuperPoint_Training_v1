"""
SuperPoint训练脚本

当前版本默认从头训练标准 65 类 detector。
如需继续训练，显式传入 --resume。

训练流程:
1. 加载数据集
2. 创建/按需加载模型
3. 迭代训练:
   - 前向传播
   - 计算损失
   - 反向传播
   - 更新参数
4. 保存模型检查点
5. 保存最终模型到checkpoints/superpoint_final.pth
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DataParallel
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config
from src.models.superpoint import SuperPoint
from src.models.losses import SuperPointLoss
from src.data.hpatches_dataset import get_hpatches_dataloader
from src.data.coco_dataset import get_coco_dataloader
from src.data.synthetic_shapes_dataset import get_synthetic_shapes_dataloader
from src.data.pseudo_label_dataset import get_pseudo_label_dataloader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练SuperPoint网络')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--dataset', type=str, default='hpatches', help='数据集名称（用于创建子文件夹）')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备 (cuda/cpu)')
    parser.add_argument('--resume', action='store_true', help='是否从 checkpoint_dir/superpoint_final.pth 继续训练')
    parser.add_argument('--init_checkpoint', type=str, default='', help='从指定 checkpoint 初始化权重，但从 epoch 0 开始训练')
    parser.add_argument('--pseudo_image_dir', type=str, default='', help='pseudo labels 训练时的真实图像目录')
    parser.add_argument('--pseudo_keypoint_dir', type=str, default='', help='pseudo labels 训练时的关键点 .npy 目录')
    parser.add_argument('--pseudo_heatmap_dir', type=str, default='', help='pseudo labels 训练时的聚合热图 .npy 目录，用于 soft supervision')
    parser.add_argument('--enable_low_light_aug', action='store_true', help='对 pseudo labels 训练图像启用低光照/低纹理退化增强')
    parser.add_argument('--soft_heatmap_weight', type=float, default=0.0, help='soft heatmap 蒸馏损失权重')
    parser.add_argument('--soft_heatmap_warmup_epochs', type=int, default=2, help='soft heatmap 蒸馏的 warmup epoch 数')
    parser.add_argument('--soft_heatmap_min_target', type=float, default=0.0, help='soft heatmap 蒸馏时忽略低于该值的 target')
    parser.add_argument('--entropy_weight', type=float, default=0.0, help='heatmap entropy 正则权重，避免过早塌缩')
    parser.add_argument('--det_weight', type=float, default=1.0, help='检测损失权重')
    parser.add_argument('--consistency_weight', type=float, default=0.0, help='两视图一致性损失权重')
    parser.add_argument('--consistency_warmup_epochs', type=int, default=3, help='一致性损失 warmup epoch 数')
    parser.add_argument('--consistency_min_confidence', type=float, default=0.05, help='一致性损失只在高置信区域计算')
    parser.add_argument('--dustbin_weight', type=float, default=None, help='覆盖 config.DUSTBIN_WEIGHT')
    parser.add_argument('--entropy_warmup_epochs', type=int, default=None, help='entropy 正则 warmup epoch 数 (默认跟 soft 同步)')
    parser.add_argument('--init_from_magicpoint', type=str, default='', help='从 MagicLeap 公开预训练权重初始化 (rpautrat/SuperPoint weights/superpoint_v6_from_tf.pth)')
    parser.add_argument('--enable_night_preprocess', action='store_true', help='pseudo_labels 训练时启用 CLAHE+gamma 提亮预处理 (与 HA 工具一致)')

    return parser.parse_args()


def create_data_loader(config, dataset_name, pseudo_image_dir='', pseudo_keypoint_dir='', pseudo_heatmap_dir='', enable_low_light_aug=False, enable_night_preprocess=False):
    """
    创建数据加载器

    Args:
        config: 配置对象
        dataset_name: 数据集名称
        pseudo_image_dir: pseudo labels 图像目录
        pseudo_keypoint_dir: pseudo labels 关键点目录

    Returns:
        train_loader: 数据加载器
    """
    print("=" * 60)
    print("创建数据集")
    print("=" * 60)
    print(f"  - 数据集名称: {dataset_name}")
    print(f"  - 图像尺寸: {config.IMAGE_HEIGHT} x {config.IMAGE_WIDTH}")
    print(f"  - 批次大小: {config.BATCH_SIZE}")

    if dataset_name == 'hpatches':
        train_loader = get_hpatches_dataloader(
            root_dir='dataset',
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            split='train',
            use_hard=True,
            image_height=config.IMAGE_HEIGHT,
            image_width=config.IMAGE_WIDTH
        )
    elif dataset_name == 'coco':
        train_loader = get_coco_dataloader(
            root_dir='dataset',
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            split='train',
            image_height=config.IMAGE_HEIGHT,
            image_width=config.IMAGE_WIDTH,
            use_synthetic_pairs=True
        )
    elif dataset_name == 'synthetic_shapes':
        train_loader = get_synthetic_shapes_dataloader(
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            split='train',
            image_height=config.IMAGE_HEIGHT,
            image_width=config.IMAGE_WIDTH,
            grid_size=config.GRID_SIZE
        )
    elif dataset_name == 'pseudo_labels':
        if not pseudo_image_dir or not pseudo_keypoint_dir:
            raise ValueError('使用 pseudo_labels 训练时，必须提供 --pseudo_image_dir 和 --pseudo_keypoint_dir')
        train_loader = get_pseudo_label_dataloader(
            image_dir=pseudo_image_dir,
            keypoint_dir=pseudo_keypoint_dir,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            image_height=config.IMAGE_HEIGHT,
            image_width=config.IMAGE_WIDTH,
            grid_size=config.GRID_SIZE,
            heatmap_dir=pseudo_heatmap_dir,
            enable_low_light_aug=enable_low_light_aug,
        )
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    print(f"  - 训练样本数: {len(train_loader.dataset)}")
    print(f"  - 训练批次数: {len(train_loader)}")
    print()

    return train_loader


def setup_device(args):
    """
    设置训练设备，支持多GPU并行

    Args:
        args: 命令行参数

    Returns:
        device: torch.device
        n_gpus: 可用的GPU数量
    """
    if args.device == 'cuda' and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"检测到 {n_gpus} 张 GPU，将使用多卡训练")
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
            if n_gpus > 1:
                print(f"  GPU 1: {torch.cuda.get_device_name(1)}")
            device = torch.device('cuda')
        else:
            print(f"使用单GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')
            n_gpus = 1
    else:
        print("使用 CPU 进行训练")
        device = torch.device('cpu')
        n_gpus = 0

    return device, n_gpus


def create_model(config, n_gpus=0):
    """
    创建SuperPoint模型

    模型结构:
    - 共享编码器: VGG风格的CNN，下采样8倍
    - 检测头: 预测每个位置是关键点的概率
    - 描述子头: 输出256维的描述子向量

    Args:
        config: 配置对象
        n_gpus: GPU数量，>1时使用多卡训练
    """
    print("=" * 60)
    print("创建SuperPoint模型")
    print("=" * 60)
    print(f"  - 编码器维度: {config.ENCODER_DIM}")
    print(f"  - 下采样倍率: {config.GRID_SIZE}")
    print(f"  - 描述子维度: 256")
    print(f"  - 检测头: Conv2d(256 -> 65)")
    print(f"  - 描述子头: Conv2d(256 -> 256)")

    model = SuperPoint(
        encoder_dim=config.ENCODER_DIM,
        grid_size=config.GRID_SIZE
    )

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数量: {trainable_params:,}")
    print()

    # 多GPU并行
    if n_gpus > 1:
        print(f"  - 使用 DataParallel 包装模型")
        model = DataParallel(model)

    return model


def load_checkpoint_weights(model, checkpoint_path, device, is_parallel=False):
    """
    从指定 checkpoint 加载权重，不恢复 epoch。

    Args:
        model: SuperPoint模型
        checkpoint_path: checkpoint 文件路径
        device: 计算设备
        is_parallel: 模型是否被DataParallel包装

    Returns:
        model: 加载后的模型
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'初始化 checkpoint 不存在: {checkpoint_path}')

    print("=" * 60)
    print("加载初始化权重")
    print("=" * 60)
    # 老架构 / MagicLeap 走专用 loader,新架构走原 state_dict 路径
    from src.models.superpoint import detect_arch_from_ckpt, load_legacy_weights, load_magicpoint_weights
    arch = detect_arch_from_ckpt(checkpoint_path)
    if arch == "legacy":
        model, _ = load_legacy_weights(model, checkpoint_path, verbose=True)
        return model
    if arch == "magicpoint":
        model = load_magicpoint_weights(model, checkpoint_path, strict=False, verbose=True)
        return model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    if is_parallel:
        final_state_dict = {}
        for k, v in new_state_dict.items():
            if not k.startswith('module.'):
                final_state_dict['module.' + k] = v
            else:
                final_state_dict[k] = v
        new_state_dict = final_state_dict

    model.load_state_dict(new_state_dict)

    loaded_epoch = checkpoint.get('epoch', 0)
    loaded_loss = checkpoint.get('loss', 0)
    print(f"  - 成功加载: {checkpoint_path}")
    print(f"  - checkpoint epoch: {loaded_epoch}")
    print(f"  - checkpoint loss: {loaded_loss:.4f}")
    print("  - 仅初始化权重，不恢复训练进度")
    print()
    return model


def load_pretrained_model(model, checkpoint_dir, device, is_parallel=False):
    """
    加载预训练模型（checkpoint_dir/superpoint_final.pth）

    Args:
        model: SuperPoint模型
        checkpoint_dir: 检查点目录
        device: 计算设备
        is_parallel: 模型是否被DataParallel包装

    Returns:
        model: 加载后的模型
        start_epoch: 起始epoch
    """
    pretrained_path = os.path.join(checkpoint_dir, 'superpoint_final.pth')

    if os.path.exists(pretrained_path):
        print("=" * 60)
        print("加载预训练模型")
        print("=" * 60)
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 处理checkpoint中的key（可能带有module.前缀或没有）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
            else:
                new_state_dict[k] = v

        # 如果模型被DataParallel包装但checkpoint没有module.前缀，需要添加
        if is_parallel:
            final_state_dict = {}
            for k, v in new_state_dict.items():
                if not k.startswith('module.'):
                    final_state_dict['module.' + k] = v
                else:
                    final_state_dict[k] = v
            new_state_dict = final_state_dict

        model.load_state_dict(new_state_dict)

        loaded_epoch = checkpoint.get('epoch', 0)
        loaded_loss = checkpoint.get('loss', 0)

        print(f"  - 成功加载: {pretrained_path}")
        print(f"  - 训练epoch: {loaded_epoch}")
        print(f"  - 损失值: {loaded_loss:.4f}")
        print()

        return model, loaded_epoch
    else:
        print("=" * 60)
        print("未找到预训练模型，将从头开始训练")
        print("=" * 60)
        print()
        return model, 0


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, print_interval=10, n_gpus=1):
    """
    训练一个epoch

    Args:
        model: SuperPoint模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        epoch: 当前epoch
        print_interval: 打印间隔
        n_gpus: GPU数量，用于处理多GPU输出
    """
    model.train()

    running_loss = 0.0
    running_det_loss = 0.0
    running_desc_loss = 0.0
    running_soft_loss = 0.0
    running_entropy_loss = 0.0
    running_consistency_loss = 0.0
    batch_count = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (images, images_consistency, soft_heatmaps, labels) in enumerate(pbar):
        # 将数据移动到设备
        images = images.to(device)
        images_consistency = images_consistency.to(device)
        soft_heatmaps = soft_heatmaps.to(device)
        labels = labels.to(device)

        # 前向传播
        descriptors, scores = model(images)

        # 计算损失
        total_loss, det_loss, desc_loss, soft_loss, entropy_loss = criterion(
            descriptors, scores, labels, valid_mask=None, soft_heatmap=soft_heatmaps, epoch=epoch
        )

        consistency_loss = scores.new_tensor(0.0)
        if hasattr(criterion, 'consistency_weight') and criterion.consistency_weight > 0.0 and epoch >= criterion.consistency_warmup_epochs:
            with torch.no_grad():
                _, consistency_teacher_scores = model(images_consistency)
            consistency_loss = criterion.consistency_weight * F.l1_loss(
                torch.softmax(scores, dim=1),
                torch.softmax(consistency_teacher_scores, dim=1),
                reduction='mean'
            )
            total_loss = total_loss + consistency_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 统计
        running_loss += total_loss.item()
        running_det_loss += det_loss.item()
        running_desc_loss += desc_loss.item()
        running_soft_loss += soft_loss.item()
        running_entropy_loss += entropy_loss.item()
        running_consistency_loss += consistency_loss.item()
        batch_count += 1

        # 更新进度条
        if batch_idx % print_interval == 0:
            pbar.set_postfix({
                'loss': f'{running_loss / batch_count:.4f}',
                'det': f'{running_det_loss / batch_count:.4f}',
                'soft': f'{running_soft_loss / batch_count:.4f}',
                'cons': f'{running_consistency_loss / batch_count:.4f}',
                'ent': f'{running_entropy_loss / batch_count:.4f}',
                'desc': f'{running_desc_loss / batch_count:.4f}'
            })

    epoch_loss = running_loss / batch_count
    epoch_det_loss = running_det_loss / batch_count
    epoch_desc_loss = running_desc_loss / batch_count

    return epoch_loss, epoch_det_loss, epoch_desc_loss


def save_checkpoint(model, optimizer, epoch, loss, dataset_dir, filename):
    """
    保存模型检查点到数据集子文件夹
    """
    os.makedirs(dataset_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    filepath = os.path.join(dataset_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"  保存检查点: {filepath}")


def main():
    """
    主训练函数

    完整的训练流程:
    1. 初始化配置、模型、数据
    2. 尝试加载预训练模型继续训练
    3. 迭代训练多个epoch
    4. 定期保存模型检查点到数据集子文件夹
    5. 保存最终模型到checkpoints/superpoint_final.pth
    """
    # 解析参数
    args = parse_args()

    # 创建设置
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.CHECKPOINT_DIR = args.checkpoint_dir

    # 数据集名称
    dataset_name = args.dataset

    # 设置设备（支持多GPU）
    device, n_gpus = setup_device(args)
    print(f"使用设备: {device}")
    if n_gpus > 1:
        print(f"并行GPU数量: {n_gpus}")
        print()

    # 创建数据加载器
    train_loader = create_data_loader(
        config,
        dataset_name,
        pseudo_image_dir=args.pseudo_image_dir,
        pseudo_keypoint_dir=args.pseudo_keypoint_dir,
        pseudo_heatmap_dir=args.pseudo_heatmap_dir,
        enable_low_light_aug=args.enable_low_light_aug,
    )

    # 创建模型（支持多GPU）
    model = create_model(config, n_gpus)
    model = model.to(device)

    # 训练初始化策略：
    # 1) --resume: 从 checkpoint_dir/superpoint_final.pth 恢复训练进度
    # 2) --init_checkpoint: 仅加载权重，从 epoch 0 开始
    # 3) 默认从头训练
    if args.resume:
        model, loaded_epoch = load_pretrained_model(
            model, config.CHECKPOINT_DIR, device, is_parallel=(n_gpus > 1)
        )
    else:
        loaded_epoch = 0
        if args.init_checkpoint:
            model = load_checkpoint_weights(
                model, args.init_checkpoint, device, is_parallel=(n_gpus > 1)
            )
        else:
            print("=" * 60)
            print("默认从头开始训练（未启用 --resume / --init_checkpoint）")
            print("=" * 60)
            print()

    # 创建损失函数和优化器
    criterion = SuperPointLoss(
        det_weight=args.det_weight,
        desc_weight=config.DESC_LOSS_WEIGHT,
        dustbin_weight=0.1,
        soft_heatmap_weight=args.soft_heatmap_weight,
        soft_heatmap_warmup_epochs=args.soft_heatmap_warmup_epochs,
        soft_heatmap_min_target=args.soft_heatmap_min_target,
        entropy_weight=args.entropy_weight,
    )
    criterion.consistency_weight = args.consistency_weight
    criterion.consistency_warmup_epochs = args.consistency_warmup_epochs

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 数据集子文件夹路径
    dataset_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, dataset_name)

    print("=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"  - 数据集: {dataset_name}")
    if dataset_name == 'pseudo_labels':
        print(f"  - pseudo 图像目录: {args.pseudo_image_dir}")
        print(f"  - pseudo 关键点目录: {args.pseudo_keypoint_dir}")
        if args.pseudo_heatmap_dir:
            print(f"  - pseudo 热图目录: {args.pseudo_heatmap_dir}")
        print(f"  - 低光增强: {'开启' if args.enable_low_light_aug else '关闭'}")
    if args.init_checkpoint:
        print(f"  - 初始化权重: {args.init_checkpoint}")
    print(f"  - 检查点目录: {dataset_checkpoint_dir}")
    print(f"  - Epochs: {config.NUM_EPOCHS}")
    print(f"  - 学习率: {config.LEARNING_RATE}")
    print(f"  - 批次大小: {config.BATCH_SIZE}")
    print(f"  - 保存间隔: {config.SAVE_INTERVAL}")
    print(f"  - detector 损失权重: {args.det_weight}")
    print(f"  - soft heatmap 蒸馏权重: {args.soft_heatmap_weight}")
    print(f"  - soft heatmap warmup epochs: {args.soft_heatmap_warmup_epochs}")
    print(f"  - soft heatmap 最小目标: {args.soft_heatmap_min_target}")
    print(f"  - entropy 正则权重: {args.entropy_weight}")
    print(f"  - consistency 权重: {args.consistency_weight}")
    print(f"  - consistency warmup epochs: {args.consistency_warmup_epochs}")
    print(f"  - consistency 最小置信度: {args.consistency_min_confidence}")
    print(f"  - 描述子损失权重: {config.DESC_LOSS_WEIGHT}（当前仅训练 detector）")
    if loaded_epoch > 0:
        print(f"  - 起始epoch: {loaded_epoch + 1}（从预训练模型继续）")
    print()

    start_time = time.time()

    # 从loaded_epoch之后开始训练
    for epoch in range(loaded_epoch + 1, loaded_epoch + config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # 训练一个epoch
        train_loss, det_loss, desc_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 学习率调度
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 打印训练信息
        print(f"Epoch {epoch}/{loaded_epoch + config.NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f} (Det: {det_loss:.4f}, Desc: {desc_loss:.4f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # 保存检查点到数据集子文件夹
        if epoch % config.SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                dataset_checkpoint_dir,
                f'superpoint_epoch_{epoch}.pth'
            )

    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("训练完成!")
    print(f"总训练时间: {total_time / 60:.2f} 分钟")
    print("=" * 60)

    # 保存最终模型到checkpoints/superpoint_final.pth（不放在数据集子文件夹）
    final_path = os.path.join(config.CHECKPOINT_DIR, 'superpoint_final.pth')
    checkpoint = {
        'epoch': loaded_epoch + config.NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'dataset': dataset_name,
    }
    torch.save(checkpoint, final_path)
    print(f"  保存最终模型: {final_path}")


if __name__ == '__main__':
    main()

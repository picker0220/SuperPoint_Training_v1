"""
SuperPoint推理与可视化脚本

用于测试训练好的模型效果，可视化检测到的关键点
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config
from src.models.superpoint import SuperPoint


def load_model(checkpoint_path, device='cuda'):
    """
    加载训练好的模型

    Args:
        checkpoint_path: 检查点文件路径
        device: 计算设备

    Returns:
        model: 加载好的模型
    """
    config = Config()
    model = SuperPoint(
        encoder_dim=config.ENCODER_DIM,
        grid_size=config.GRID_SIZE
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']

        # 处理多GPU训练的模型（DataParallel会在key前加module.）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        print(f"成功加载检查点: {checkpoint_path}")
        print(f"训练epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"损失值: {checkpoint.get('loss', 'unknown')}")
    else:
        print(f"警告: 检查点文件不存在 {checkpoint_path}")
        print("使用随机初始化的模型")

    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image, target_height=480, target_width=640):
    """
    预处理图像

    Args:
        image: 输入图像 (numpy array)
        target_height: 目标高度
        target_width: 目标宽度

    Returns:
        image_tensor: 预处理后的图像张量
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (target_width, target_height))

    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    return image_tensor


def detect_keypoints(model, image_tensor, device, threshold=0.005, nms_distance=4):
    """
    使用标准 SuperPoint 65 类输出检测关键点

    Args:
        model: SuperPoint模型
        image_tensor: 输入图像张量 [1, 1, H, W]
        device: 计算设备
        threshold: 关键点检测阈值（full-resolution heatmap 上）
        nms_distance: NMS最小距离阈值

    Returns:
        keypoints: 检测到的关键点列表 [(x, y), ...]
        scores: 关键点对应的分数
        heatmap: full-resolution 概率热力图 [H, W]
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        descriptors, scores = model(image_tensor)

    # scores: [1, 65, H/8, W/8]
    prob = torch.softmax(scores, dim=1)[:, :-1, :, :]  # [1, 64, H/8, W/8]
    heatmap = torch.pixel_shuffle(prob, upscale_factor=8)  # [1, 1, H, W]
    heatmap = heatmap.squeeze().cpu().numpy()

    print(f"heatmap stats -> min: {heatmap.min():.6f}, max: {heatmap.max():.6f}, mean: {heatmap.mean():.6f}, threshold: {threshold}")

    ys, xs = np.where(heatmap > threshold)
    candidates = [(float(heatmap[y, x]), int(x), int(y)) for y, x in zip(ys, xs)]
    candidates.sort(reverse=True)

    keypoints = []
    keypoint_scores = []
    for score, x, y in candidates:
        too_close = False
        for kx, ky in keypoints:
            dist = ((x - kx) ** 2 + (y - ky) ** 2) ** 0.5
            if dist < nms_distance:
                too_close = True
                break

        if not too_close:
            keypoints.append((x, y))
            keypoint_scores.append(score)

    return keypoints, keypoint_scores, heatmap


def visualize_results(image, keypoints, heatmap=None, title="SuperPoint Detection", save_path=None):
    """
    可视化检测结果

    Args:
        image: 输入图像
        keypoints: 检测到的关键点列表
        heatmap: 预测热力图（可选）
        title: 图像标题
        save_path: 保存路径（可选）
    """
    if len(image.shape) == 2:
        image_display = np.stack([image] * 3, axis=-1)
    else:
        image_display = image.copy()

    fig, axes = plt.subplots(1, 2 if heatmap is not None else 1, figsize=(12, 6))

    if heatmap is not None:
        # 原始图像 + 关键点
        axes[0].imshow(image_display)
        for x, y in keypoints:
            circle = Circle((x, y), radius=2, color='lime', linewidth=0)
            axes[0].add_patch(circle)
        axes[0].set_title(f'{title}\nDetected {len(keypoints)} keypoints')
        axes[0].axis('off')

        # heatmap 已经是原图分辨率
        im = axes[1].imshow(heatmap, cmap='hot')
        axes[1].set_title('Predicted Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
    else:
        axes.imshow(image_display)
        for x, y in keypoints:
            circle = Circle((x, y), radius=2, color='lime', linewidth=0)
            axes.add_patch(circle)
        axes.set_title(f'{title}\nDetected {len(keypoints)} keypoints')
        axes.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"结果已保存: {save_path}")
    else:
        plt.savefig('visualization_result.png')
        print(f"结果已保存: visualization_result.png")

    plt.close()


def detect_on_image_file(model, image_path, device='cuda', threshold=0.005, save_path=None):
    """
    对图像文件进行关键点检测

    Args:
        model: SuperPoint模型
        image_path: 图像文件路径
        device: 计算设备
        threshold: 关键点检测阈值
        save_path: 结果保存路径

    Returns:
        keypoints: 检测到的关键点列表
    """
    print(f"\n处理图像: {image_path}")

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"图像尺寸: {image.shape}")

    # 预处理
    image_tensor = preprocess_image(image)

    # 检测关键点
    keypoints, scores, heatmap = detect_keypoints(model, image_tensor, device, threshold)

    print(f"检测到 {len(keypoints)} 个关键点")

    # 可视化
    visualize_results(image_rgb, keypoints, heatmap, title=os.path.basename(image_path), save_path=save_path)

    return keypoints


def detect_on_hpatches_sequence(model, hpatches_root, sequence_name='i_ajuntament', device='cuda', threshold=0.005):
    """
    在HPatches序列上测试模型

    Args:
        model: SuperPoint模型
        hpatches_root: HPatches数据集根目录
        sequence_name: 序列名称 (如 'i_ajuntament', 'v_abstract')
        device: 计算设备
        threshold: 关键点检测阈值
    """
    import glob

    # 查找HPatches目录
    possible_paths = [
        os.path.join(hpatches_root, 'hpatches', 'hpatches-release', sequence_name),
        os.path.join(hpatches_root, 'hpatches-release', sequence_name),
        os.path.join(hpatches_root, sequence_name),
    ]

    seq_path = None
    for p in possible_paths:
        if os.path.exists(p):
            seq_path = p
            break

    if seq_path is None:
        print(f"序列不存在: {sequence_name}")
        print("可用序列:")
        hpatches_dir = os.path.join(hpatches_root, 'hpatches', 'hpatches-release')
        if os.path.exists(hpatches_dir):
            for name in sorted(os.listdir(hpatches_dir))[:10]:
                print(f"  - {name}")
        return None

    ref_path = os.path.join(seq_path, 'ref.png')
    if not os.path.exists(ref_path):
        print(f"ref.png 不存在于 {seq_path}")
        return None

    print(f"\n测试HPatches序列: {sequence_name}")
    print(f"路径: {seq_path}")

    # 读取参考图像
    image = cv2.imread(ref_path)
    if image is None:
        print(f"无法读取图像: {ref_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"图像尺寸: {image.shape}")

    # 预处理
    image_tensor = preprocess_image(image)

    # 检测关键点
    keypoints, scores, heatmap = detect_keypoints(model, image_tensor, device, threshold)

    print(f"检测到 {len(keypoints)} 个关键点")

    # 可视化
    visualize_results(image_rgb, keypoints, heatmap, title=f"HPatches: {sequence_name}",
                     save_path=f"hpatches_test_{sequence_name}.png")

    return keypoints


def detect_on_hpatches_random(model, hpatches_root, device='cuda', threshold=0.3, num_sequences=5):
    """
    在随机HPatches序列上测试模型

    Args:
        model: SuperPoint模型
        hpatches_root: HPatches数据集根目录
        device: 计算设备
        threshold: 关键点检测阈值
        num_sequences: 测试的序列数量
    """
    import random

    # 查找HPatches目录
    possible_paths = [
        os.path.join(hpatches_root, 'hpatches', 'hpatches-release'),
        os.path.join(hpatches_root, 'hpatches-release'),
    ]

    hpatches_dir = None
    for p in possible_paths:
        if os.path.exists(p):
            hpatches_dir = p
            break

    if hpatches_dir is None:
        print("无法找到HPatches数据集目录")
        return

    # 获取所有序列
    sequences = [d for d in os.listdir(hpatches_dir) if os.path.isdir(os.path.join(hpatches_dir, d))]
    sequences = sorted(sequences)

    print(f"找到 {len(sequences)} 个HPatches序列")
    print(f"随机测试 {num_sequences} 个序列...\n")

    # 随机选择序列
    random.seed(42)
    selected = random.sample(sequences, min(num_sequences, len(sequences)))

    for seq_name in selected:
        keypoints = detect_on_hpatches_sequence(
            model, hpatches_root, seq_name, device, threshold
        )
        if keypoints is not None:
            print(f"  -> 检测到 {len(keypoints)} 个关键点\n")


def detect_on_coco_image(model, coco_root, image_id=None, device='cuda', threshold=0.3):
    """
    在COCO图像上测试模型

    Args:
        model: SuperPoint模型
        coco_root: COCO数据集根目录
        image_id: 图像ID（可选，None则随机选择）
        device: 计算设备
        threshold: 关键点检测阈值
    """
    import random

    train_dir = os.path.join(coco_root, 'train2017')
    val_dir = os.path.join(coco_root, 'val2017')

    image_dir = train_dir if os.path.exists(train_dir) else val_dir
    if not os.path.exists(image_dir):
        print(f"COCO图像目录不存在: {image_dir}")
        return None

    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    if not images:
        print("未找到COCO图像")
        return None

    if image_id is None:
        random.seed(42)
        image_name = random.choice(images)
    else:
        image_name = f"{image_id:012d}.jpg"
        if image_name not in images:
            image_name = f"COCO_val2017_{image_id:08d}.jpg"

    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        print(f"图像不存在: {image_path}")
        return None

    print(f"\n测试COCO图像: {image_name}")

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"图像尺寸: {image.shape}")

    # 预处理
    image_tensor = preprocess_image(image)

    # 检测关键点
    keypoints, scores, heatmap = detect_keypoints(model, image_tensor, device, threshold)

    print(f"检测到 {len(keypoints)} 个关键点")

    # 可视化
    visualize_results(image_rgb, keypoints, heatmap, title=f"COCO: {image_name}",
                     save_path=f"coco_test_{image_name}.png")

    return keypoints


def detect_on_synthetic_image(model, device='cuda'):
    """
    在合成图像上测试模型（用于快速验证模型是否工作）
    """
    print("\n在合成图像上测试模型...")

    # 创建简单的测试图像（网格点）
    image = np.zeros((480, 640), dtype=np.uint8)
    grid_size = 40

    # 绘制网格关键点
    for y in range(grid_size, 480 - grid_size, grid_size):
        for x in range(grid_size, 640 - grid_size, grid_size):
            cv2.circle(image, (x, y), 5, 255, -1)

    # 添加一些噪声
    noise = np.random.normal(0, 15, image.shape).astype(np.float32)
    image = image.astype(np.float32) + noise
    image = np.clip(image, 0, 255).astype(np.uint8)

    image_tensor = preprocess_image(image)
    keypoints, scores, heatmap = detect_keypoints(model, image_tensor, device, threshold=0.3)

    print(f"合成图像检测到 {len(keypoints)} 个关键点")

    # 可视化
    image_rgb = np.stack([image] * 3, axis=-1)
    visualize_results(image_rgb, keypoints, heatmap, title="Synthetic Image Test", save_path="synthetic_test.png")

    return keypoints


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SuperPoint关键点检测可视化')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/superpoint_final.pth',
                        help='模型检查点路径')
    parser.add_argument('--image', type=str, default=None,
                        help='要检测的图像路径')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='关键点检测阈值 (0-1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--save', type=str, default=None,
                        help='结果保存路径')
    parser.add_argument('--coco', type=str, default=None,
                        help='COCO数据集路径')
    parser.add_argument('--hpatches', type=str, default=None,
                        help='HPatches数据集路径（测试HPatches序列）')
    parser.add_argument('--hpatches_seq', type=str, default=None,
                        help='HPatches序列名称（如 i_ajuntament）')
    parser.add_argument('--hpatches_random', action='store_true',
                        help='在随机HPatches序列上测试')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model(args.checkpoint, device)

    if args.coco:
        # 测试COCO图像
        detect_on_coco_image(model, args.coco, device=device, threshold=args.threshold)
    elif args.hpatches:
        # 测试HPatches序列
        if args.hpatches_seq:
            detect_on_hpatches_sequence(model, args.hpatches, args.hpatches_seq, device, args.threshold)
        elif args.hpatches_random:
            detect_on_hpatches_random(model, args.hpatches, device, args.threshold)
        else:
            print("请使用 --hpatches_seq 指定序列名 或 --hpatches_random 随机测试")
    elif args.image:
        # 检测指定图像
        detect_on_image_file(model, args.image, device, args.threshold, args.save)
    else:
        # 在合成图像上测试
        detect_on_synthetic_image(model, device)


if __name__ == '__main__':
    main()
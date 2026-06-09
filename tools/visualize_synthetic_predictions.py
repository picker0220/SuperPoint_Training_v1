"""
可视化 synthetic shapes 上的 GT 与预测结果对比

功能：
1. 从 SyntheticShapesDataset 随机抽样若干张图
2. 加载训练好的 detector checkpoint
3. 在每张图上做关键点检测
4. 生成三联图：原图+GT、原图+Pred、Pred Heatmap
5. 保存到 outputs/visualizations/synthetic_shapes_pred/
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.synthetic_shapes_dataset import SyntheticShapesDataset
from visualize import load_model, detect_keypoints


def parse_args():
    parser = argparse.ArgumentParser(description='可视化 synthetic shapes 上的 GT 与预测')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--num_samples', type=int, default=5, help='抽样数量')
    parser.add_argument('--threshold', type=float, default=0.005, help='检测阈值')
    parser.add_argument('--device', type=str, default='cuda', help='设备 cuda/cpu')
    return parser.parse_args()


def decode_gt_keypoints(labels, grid_size=8):
    keypoints = []
    h, w = labels.shape
    for gy in range(h):
        for gx in range(w):
            cls = int(labels[gy, gx])
            if cls == 64:
                continue
            cell_y = cls // grid_size
            cell_x = cls % grid_size
            x = gx * grid_size + cell_x
            y = gy * grid_size + cell_y
            keypoints.append((x, y))
    return keypoints


def draw_points(ax, image, keypoints, color='lime', radius=2, title=''):
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    for x, y in keypoints:
        circle = Circle((x, y), radius=radius, color=color, linewidth=0)
        ax.add_patch(circle)
    ax.set_title(title)
    ax.axis('off')


def main():
    args = parse_args()

    device = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device=device)

    dataset = SyntheticShapesDataset(num_samples=args.num_samples, image_height=480, image_width=640, grid_size=8)

    out_dir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        'outputs', 'visualizations', 'synthetic_shapes_pred'
    )
    os.makedirs(out_dir, exist_ok=True)

    for i in range(args.num_samples):
        image_tensor, _, label_tensor, _, _ = dataset[i]
        image = (image_tensor.squeeze().numpy() * 255).astype(np.uint8)
        gt_keypoints = decode_gt_keypoints(label_tensor.numpy(), grid_size=8)

        input_tensor = image_tensor.unsqueeze(0)  # [1, 1, H, W]
        pred_keypoints, pred_scores, heatmap = detect_keypoints(
            model, input_tensor, device, threshold=args.threshold, nms_distance=4
        )

        print(f'sample_{i}: GT={len(gt_keypoints)}, Pred={len(pred_keypoints)}')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        draw_points(axes[0], image, gt_keypoints, color='cyan', radius=2, title=f'GT ({len(gt_keypoints)})')
        draw_points(axes[1], image, pred_keypoints, color='lime', radius=2, title=f'Pred ({len(pred_keypoints)})')
        im = axes[2].imshow(heatmap, cmap='hot')
        axes[2].set_title('Pred Heatmap')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        plt.tight_layout()

        save_path = os.path.join(out_dir, f'synthetic_pred_{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f'已保存: {save_path}')

    print(f'全部结果保存在: {out_dir}')


if __name__ == '__main__':
    main()

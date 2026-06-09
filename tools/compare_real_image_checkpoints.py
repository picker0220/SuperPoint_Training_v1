"""
对比真实图像在不同 checkpoint 下的关键点检测结果。

用途：
1. 对比 synthetic teacher
2. 对比 pseudo labels finetune 的 epoch 5 / epoch 10
3. 输出一张多列对比图，便于直观看哪个 checkpoint 更干净、更稳定
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualize import load_model, preprocess_image, detect_keypoints


def parse_args():
    parser = argparse.ArgumentParser(description='对比多个 checkpoint 在真实图像上的检测结果')
    parser.add_argument('--image', type=str, required=True, help='输入真实图像路径')
    parser.add_argument('--teacher_ckpt', type=str, required=True, help='synthetic teacher checkpoint')
    parser.add_argument('--pseudo_epoch5_ckpt', type=str, required=True, help='pseudo labels finetune epoch 5 checkpoint')
    parser.add_argument('--pseudo_epoch10_ckpt', type=str, required=True, help='pseudo labels finetune epoch 10 checkpoint')
    parser.add_argument('--threshold', type=float, default=0.03, help='关键点阈值')
    parser.add_argument('--nms_distance', type=int, default=10, help='NMS 距离')
    parser.add_argument('--device', type=str, default='cuda', help='设备 cuda/cpu')
    parser.add_argument('--save', type=str, default='', help='保存路径，默认输出到 outputs/visualizations/checkpoint_compare/')
    return parser.parse_args()


def draw_keypoints(ax, image_rgb, keypoints, title):
    ax.imshow(image_rgb)
    for x, y in keypoints:
        circle = Circle((x, y), radius=2, color='lime', linewidth=0)
        ax.add_patch(circle)
    ax.set_title(f'{title}\nDetected {len(keypoints)} keypoints')
    ax.axis('off')


def draw_heatmap(ax, heatmap, title):
    im = ax.imshow(heatmap, cmap='hot')
    ax.set_title(title)
    ax.axis('off')
    return im


def infer_one(checkpoint_path, image_bgr, device, threshold, nms_distance):
    model = load_model(checkpoint_path, device=device)
    image_tensor = preprocess_image(image_bgr)
    keypoints, scores, heatmap = detect_keypoints(
        model, image_tensor, device=device, threshold=threshold, nms_distance=nms_distance
    )
    return keypoints, heatmap


def main():
    args = parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f'无法读取图像: {args.image}')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    teacher_kp, teacher_heatmap = infer_one(
        args.teacher_ckpt, image_bgr, args.device, args.threshold, args.nms_distance
    )
    pseudo5_kp, pseudo5_heatmap = infer_one(
        args.pseudo_epoch5_ckpt, image_bgr, args.device, args.threshold, args.nms_distance
    )
    pseudo10_kp, pseudo10_heatmap = infer_one(
        args.pseudo_epoch10_ckpt, image_bgr, args.device, args.threshold, args.nms_distance
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    draw_keypoints(axes[0, 0], image_rgb, teacher_kp, 'Teacher')
    draw_keypoints(axes[0, 1], image_rgb, pseudo5_kp, 'Pseudo Epoch 5')
    draw_keypoints(axes[0, 2], image_rgb, pseudo10_kp, 'Pseudo Epoch 10')

    im0 = draw_heatmap(axes[1, 0], teacher_heatmap, 'Teacher Heatmap')
    im1 = draw_heatmap(axes[1, 1], pseudo5_heatmap, 'Pseudo Epoch 5 Heatmap')
    im2 = draw_heatmap(axes[1, 2], pseudo10_heatmap, 'Pseudo Epoch 10 Heatmap')

    fig.colorbar(im0, ax=axes[1, 0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046)
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()

    if args.save:
        save_path = args.save
    else:
        out_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            'outputs', 'visualizations', 'checkpoint_compare'
        )
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.image))[0]
        save_path = os.path.join(out_dir, f'{stem}_compare.png')

    plt.savefig(save_path)
    plt.close()
    print(f'对比结果已保存: {save_path}')


if __name__ == '__main__':
    main()

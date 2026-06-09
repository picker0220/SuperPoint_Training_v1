"""
批量对比多个 checkpoint 在一组真实图像上的检测结果。

功能：
- 读取一个真实图像目录中的多张图
- 对每张图分别比较 teacher / pseudo epoch5 / pseudo epoch10
- 为每张图输出一张对比图
- 便于快速抽检一小批图像的整体质量
"""

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.compare_real_image_checkpoints import infer_one, draw_keypoints, draw_heatmap
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='批量对比多个 checkpoint 在真实图像上的检测结果')
    parser.add_argument('--image_dir', type=str, required=True, help='真实图像目录')
    parser.add_argument('--teacher_ckpt', type=str, required=True, help='synthetic teacher checkpoint')
    parser.add_argument('--pseudo_epoch5_ckpt', type=str, required=True, help='pseudo labels finetune epoch 5 checkpoint')
    parser.add_argument('--pseudo_epoch10_ckpt', type=str, required=True, help='pseudo labels finetune epoch 10 checkpoint')
    parser.add_argument('--num_images', type=int, default=10, help='抽检图像数量')
    parser.add_argument('--threshold', type=float, default=0.03, help='关键点阈值')
    parser.add_argument('--nms_distance', type=int, default=10, help='NMS 距离')
    parser.add_argument('--device', type=str, default='cuda', help='设备 cuda/cpu')
    parser.add_argument('--output_dir', type=str, default='', help='输出目录，默认 outputs/visualizations/checkpoint_compare_batch/')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f'图像目录不存在: {args.image_dir}')

    images = [f for f in sorted(os.listdir(args.image_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not images:
        raise RuntimeError(f'目录中没有找到图像: {args.image_dir}')

    selected = images[: min(args.num_images, len(images))]

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            'outputs', 'visualizations', 'checkpoint_compare_batch'
        )
    os.makedirs(out_dir, exist_ok=True)

    print(f'共选取 {len(selected)} 张图像进行批量对比，输出目录: {out_dir}')

    for name in selected:
        image_path = os.path.join(args.image_dir, name)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f'跳过无法读取图像: {image_path}')
            continue
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

        stem = os.path.splitext(name)[0]
        save_path = os.path.join(out_dir, f'{stem}_compare.png')
        plt.savefig(save_path)
        plt.close()
        print(f'已保存: {save_path}')

    print('批量对比完成')


if __name__ == '__main__':
    main()

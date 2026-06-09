"""
生成 Homographic Adaptation 伪标签（初版）

当前版本流程：
1. 加载训练好的初始 detector checkpoint
2. 读取指定目录下的真实图像
3. 对每张图像做多次随机单应变换
4. 在每次变换图像上检测关键点热图
5. 将热图逆变换回原图坐标系并累积
6. 阈值化 + NMS，导出关键点坐标

输出目录结构：
outputs/pseudo_labels/<export_name>/
  ├─ keypoints/            # .npy 原始关键点坐标
  ├─ keypoints_txt/        # .txt 可直接查看的关键点坐标
  ├─ heatmaps/             # .npy 原始热图
  ├─ heatmaps_png/         # .png 可直接查看的热图
  ├─ overlays/             # 关键点叠加图
  └─ stats.csv             # 每张图的关键统计信息
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from src.models.superpoint import SuperPoint
from visualize import preprocess_image


def parse_args():
    parser = argparse.ArgumentParser(description='生成 homographic adaptation 伪标签')
    parser.add_argument('--checkpoint', type=str, required=True, help='初始 detector checkpoint 路径')
    parser.add_argument('--image_dir', type=str, required=True, help='输入真实图像目录')
    parser.add_argument('--export_name', type=str, default='pseudo_labels_v1', help='输出子目录名称')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--num_homographies', type=int, default=20, help='每张图像采样的随机单应数量')
    parser.add_argument('--threshold', type=float, default=0.015, help='聚合热图阈值')
    parser.add_argument('--nms_distance', type=int, default=8, help='NMS 最小距离')
    parser.add_argument('--max_images', type=int, default=0, help='最多处理多少张图，0 表示全部处理')
    parser.add_argument('--enable_low_light_consistency', action='store_true', help='额外对暗化/噪声/模糊视图做一致性聚合，适合夜间图像')
    parser.add_argument('--num_low_light_views', type=int, default=3, help='每张图额外采样多少个低光退化视图')
    parser.add_argument('--low_light_weight', type=float, default=0.7, help='低光退化视图热图在聚合中的单视图权重')
    parser.add_argument('--normalize_heatmap', action='store_true', help='聚合后将热图按最大值归一化到 [0,1]，便于不同曝光图统一阈值')
    parser.add_argument('--enable_night_preprocess', action='store_true', help='对输入图像做夜间预处理，提升暗区结构可见性')
    parser.add_argument('--clahe_clip_limit', type=float, default=2.5, help='夜间预处理时 CLAHE 的 clip limit')
    parser.add_argument('--clahe_grid_size', type=int, default=8, help='夜间预处理时 CLAHE 的网格大小')
    parser.add_argument('--gamma_lift', type=float, default=0.8, help='夜间预处理时提亮 gamma，<1 会提亮暗部')
    parser.add_argument('--enable_denoise', action='store_true', help='夜间预处理时启用轻度去噪')
    parser.add_argument('--suppress_bright_regions', action='store_true', help='对过亮区域做热图抑制，避免亮点主导 keypoint 排序')
    parser.add_argument('--bright_percentile', type=float, default=92.0, help='亮区抑制所用亮度分位数阈值')
    parser.add_argument('--bright_suppression_strength', type=float, default=0.45, help='亮区抑制强度，越大对亮区惩罚越强')
    parser.add_argument('--grid_topk', type=int, default=0, help='可选，每个网格最多保留多少个候选点，0 表示关闭')
    parser.add_argument('--grid_rows', type=int, default=6, help='grid_topk 开启时的网格行数')
    parser.add_argument('--grid_cols', type=int, default=8, help='grid_topk 开启时的网格列数')
    return parser.parse_args()


def load_model(checkpoint_path, device='cuda'):
    config = Config()
    model = SuperPoint(
        encoder_dim=config.ENCODER_DIM,
        grid_size=config.GRID_SIZE
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    return model


def random_homography(width, height, max_offset=32):
    src = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])
    dst = src.copy()
    dst += np.random.uniform(-max_offset, max_offset, size=dst.shape).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = np.linalg.inv(H)
    return H, H_inv


def apply_night_preprocess(image_bgr, clahe_clip_limit=2.5, clahe_grid_size=8,
                           gamma_lift=0.8, enable_denoise=False):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if enable_denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)

    gray_float = gray.astype(np.float32) / 255.0
    gray_float = np.power(np.clip(gray_float, 0.0, 1.0), gamma_lift)
    gray_lifted = np.clip(gray_float * 255.0, 0.0, 255.0).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(clahe_clip_limit)),
        tileGridSize=(max(1, int(clahe_grid_size)), max(1, int(clahe_grid_size)))
    )
    enhanced = clahe.apply(gray_lifted)

    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr


def detect_heatmap(model, image_bgr, device):
    image_tensor = preprocess_image(image_bgr)
    with torch.no_grad():
        _, scores = model(image_tensor.to(device))
    prob = torch.softmax(scores, dim=1)[:, :-1, :, :]
    heatmap = torch.pixel_shuffle(prob, upscale_factor=8)
    heatmap = heatmap.squeeze().cpu().numpy()
    return heatmap


def apply_low_light_degradation(image_bgr):
    image = image_bgr.astype(np.float32) / 255.0

    gamma = np.random.uniform(1.4, 3.2)
    image = np.power(np.clip(image, 0.0, 1.0), gamma)

    brightness_scale = np.random.uniform(0.4, 0.85)
    contrast_scale = np.random.uniform(0.65, 1.0)
    image = np.clip((image - 0.5) * contrast_scale + 0.5, 0.0, 1.0)
    image = np.clip(image * brightness_scale, 0.0, 1.0)

    if np.random.rand() < 0.8:
        noise_std = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0.0, noise_std, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    if np.random.rand() < 0.5:
        ksize = int(np.random.choice([3, 5]))
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=np.random.uniform(0.6, 1.8))

    if np.random.rand() < 0.35:
        shadow = np.ones_like(image, dtype=np.float32)
        x0 = np.random.randint(0, image.shape[1] // 2)
        y0 = np.random.randint(0, image.shape[0] // 2)
        x1 = np.random.randint(image.shape[1] // 2, image.shape[1])
        y1 = np.random.randint(image.shape[0] // 2, image.shape[0])
        shadow[y0:y1, x0:x1, :] *= np.random.uniform(0.35, 0.75)
        image = np.clip(image * shadow, 0.0, 1.0)

    return (image * 255.0).astype(np.uint8)


def suppress_bright_region_heatmap(heatmap, image_bgr, bright_percentile=92.0, strength=0.45):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    threshold = np.percentile(gray, np.clip(bright_percentile, 50.0, 99.9))
    bright_mask = gray >= threshold

    if not np.any(bright_mask):
        return heatmap

    gray_norm = gray / 255.0
    suppression = np.ones_like(heatmap, dtype=np.float32)
    bright_strength = np.clip((gray_norm - threshold / 255.0) / max(1e-6, 1.0 - threshold / 255.0), 0.0, 1.0)
    suppression_factor = 1.0 - np.clip(strength, 0.0, 0.95) * bright_strength
    suppression[bright_mask] = suppression_factor[bright_mask]

    suppressed = heatmap * suppression
    return suppressed.astype(np.float32)


def select_candidates_with_grid(candidates, width, height, grid_rows=6, grid_cols=8, grid_topk=0):
    if grid_topk <= 0:
        return candidates

    cell_h = max(1, height // max(1, grid_rows))
    cell_w = max(1, width // max(1, grid_cols))
    grouped = {}

    for score, x, y in candidates:
        gy = min(grid_rows - 1, y // cell_h)
        gx = min(grid_cols - 1, x // cell_w)
        grouped.setdefault((gy, gx), []).append((score, x, y))

    selected = []
    for _, items in grouped.items():
        items.sort(reverse=True)
        selected.extend(items[:grid_topk])

    selected.sort(reverse=True)
    return selected


def nms_points(heatmap, threshold=0.015, nms_distance=8, grid_topk=0, grid_rows=6, grid_cols=8):
    ys, xs = np.where(heatmap > threshold)
    candidates = [(float(heatmap[y, x]), int(x), int(y)) for y, x in zip(ys, xs)]
    candidates.sort(reverse=True)
    candidates = select_candidates_with_grid(
        candidates,
        width=heatmap.shape[1],
        height=heatmap.shape[0],
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        grid_topk=grid_topk,
    )

    keypoints = []
    for score, x, y in candidates:
        keep = True
        for kx, ky in keypoints:
            if ((x - kx) ** 2 + (y - ky) ** 2) ** 0.5 < nms_distance:
                keep = False
                break
        if keep:
            keypoints.append((x, y))
    return keypoints


def save_overlay(image_rgb, keypoints, save_path):
    overlay = image_rgb.copy()
    for x, y in keypoints:
        cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_heatmap_png(heatmap, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar(fraction=0.046)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def save_keypoints_txt(keypoints, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('x,y\n')
        for x, y in keypoints:
            f.write(f'{x},{y}\n')


def summarize_keypoints(keypoints, width, height):
    if len(keypoints) == 0:
        return 0.0, 0.0, 0.0, 0.0

    xs = np.array([x for x, _ in keypoints], dtype=np.float32)
    ys = np.array([y for _, y in keypoints], dtype=np.float32)

    left_ratio = float(np.mean(xs < width * 0.25))
    right_ratio = float(np.mean(xs > width * 0.75))
    top_ratio = float(np.mean(ys < height * 0.25))
    bottom_ratio = float(np.mean(ys > height * 0.75))
    return left_ratio, right_ratio, top_ratio, bottom_ratio


def main():
    args = parse_args()

    device = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device=device)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    export_root = os.path.join(project_root, 'outputs', 'pseudo_labels', args.export_name)
    keypoint_dir = os.path.join(export_root, 'keypoints')
    keypoint_txt_dir = os.path.join(export_root, 'keypoints_txt')
    heatmap_dir = os.path.join(export_root, 'heatmaps')
    heatmap_png_dir = os.path.join(export_root, 'heatmaps_png')
    overlay_dir = os.path.join(export_root, 'overlays')
    os.makedirs(keypoint_dir, exist_ok=True)
    os.makedirs(keypoint_txt_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(heatmap_png_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    image_paths = []
    for name in sorted(os.listdir(args.image_dir)):
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths.append(os.path.join(args.image_dir, name))

    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print(f'待处理图像数: {len(image_paths)}')
    print(f'低光一致性: {"开启" if args.enable_low_light_consistency else "关闭"}')
    if args.enable_low_light_consistency:
        print(f'  - 低光视图数: {args.num_low_light_views}')
        print(f'  - 低光视图权重: {args.low_light_weight}')
    print(f'夜间预处理: {"开启" if args.enable_night_preprocess else "关闭"}')
    if args.enable_night_preprocess:
        print(f'  - CLAHE clip limit: {args.clahe_clip_limit}')
        print(f'  - CLAHE grid size: {args.clahe_grid_size}')
        print(f'  - gamma lift: {args.gamma_lift}')
        print(f'  - 去噪: {"开启" if args.enable_denoise else "关闭"}')
    print(f'亮区抑制: {"开启" if args.suppress_bright_regions else "关闭"}')
    if args.suppress_bright_regions:
        print(f'  - 亮度分位数: {args.bright_percentile}')
        print(f'  - 抑制强度: {args.bright_suppression_strength}')
    print(f'网格限额: {args.grid_topk if args.grid_topk > 0 else "关闭"}')
    print(f'聚合后归一化: {"开启" if args.normalize_heatmap else "关闭"}')

    stats_rows = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f'跳过无法读取图像: {image_path}')
            continue

        image = cv2.resize(image, (640, 480))
        if args.enable_night_preprocess:
            image_for_detection = apply_night_preprocess(
                image,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_grid_size=args.clahe_grid_size,
                gamma_lift=args.gamma_lift,
                enable_denoise=args.enable_denoise,
            )
        else:
            image_for_detection = image.copy()

        h, w = image.shape[:2]
        agg_heatmap = np.zeros((h, w), dtype=np.float32)
        total_weight = 0.0

        base_heatmap = detect_heatmap(model, image_for_detection, device)
        agg_heatmap += base_heatmap
        total_weight += 1.0

        for _ in range(args.num_homographies):
            H, H_inv = random_homography(w, h)
            warped = cv2.warpPerspective(image_for_detection, H, (w, h), borderMode=cv2.BORDER_REFLECT)
            warped_heatmap = detect_heatmap(model, warped, device)
            back_heatmap = cv2.warpPerspective(warped_heatmap, H_inv, (w, h), flags=cv2.INTER_LINEAR)
            agg_heatmap += back_heatmap
            total_weight += 1.0

        if args.enable_low_light_consistency:
            for _ in range(args.num_low_light_views):
                degraded = apply_low_light_degradation(image_for_detection)
                degraded_heatmap = detect_heatmap(model, degraded, device)
                agg_heatmap += degraded_heatmap * args.low_light_weight
                total_weight += args.low_light_weight

                H, H_inv = random_homography(w, h)
                degraded_warped = cv2.warpPerspective(degraded, H, (w, h), borderMode=cv2.BORDER_REFLECT)
                degraded_warped_heatmap = detect_heatmap(model, degraded_warped, device)
                degraded_back_heatmap = cv2.warpPerspective(degraded_warped_heatmap, H_inv, (w, h), flags=cv2.INTER_LINEAR)
                agg_heatmap += degraded_back_heatmap * args.low_light_weight
                total_weight += args.low_light_weight

        agg_heatmap /= max(total_weight, 1e-6)
        if args.normalize_heatmap:
            max_val = float(agg_heatmap.max())
            if max_val > 1e-6:
                agg_heatmap = agg_heatmap / max_val

        if args.suppress_bright_regions:
            agg_heatmap = suppress_bright_region_heatmap(
                agg_heatmap,
                image,
                bright_percentile=args.bright_percentile,
                strength=args.bright_suppression_strength,
            )
            if args.normalize_heatmap:
                max_val = float(agg_heatmap.max())
                if max_val > 1e-6:
                    agg_heatmap = agg_heatmap / max_val

        keypoints = nms_points(
            agg_heatmap,
            threshold=args.threshold,
            nms_distance=args.nms_distance,
            grid_topk=args.grid_topk,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )

        heatmap_min = float(agg_heatmap.min())
        heatmap_max = float(agg_heatmap.max())
        heatmap_mean = float(agg_heatmap.mean())
        left_ratio, right_ratio, top_ratio, bottom_ratio = summarize_keypoints(keypoints, w, h)

        stem = os.path.splitext(os.path.basename(image_path))[0]
        np.save(os.path.join(heatmap_dir, f'{stem}.npy'), agg_heatmap)
        np.save(os.path.join(keypoint_dir, f'{stem}.npy'), np.array(keypoints, dtype=np.int32))
        save_keypoints_txt(keypoints, os.path.join(keypoint_txt_dir, f'{stem}.txt'))
        save_heatmap_png(agg_heatmap, os.path.join(heatmap_png_dir, f'{stem}.png'))
        save_overlay(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), keypoints, os.path.join(overlay_dir, f'{stem}.png'))

        stats_rows.append({
            'image': stem,
            'num_keypoints': len(keypoints),
            'heatmap_min': heatmap_min,
            'heatmap_max': heatmap_max,
            'heatmap_mean': heatmap_mean,
            'left_ratio': left_ratio,
            'right_ratio': right_ratio,
            'top_ratio': top_ratio,
            'bottom_ratio': bottom_ratio,
        })

        print(f'{stem}: {len(keypoints)} keypoints | heatmap max={heatmap_max:.4f}, mean={heatmap_mean:.6f} | total_weight={total_weight:.2f}')

    stats_csv_path = os.path.join(export_root, 'stats.csv')
    with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['image', 'num_keypoints', 'heatmap_min', 'heatmap_max', 'heatmap_mean',
                        'left_ratio', 'right_ratio', 'top_ratio', 'bottom_ratio']
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    if stats_rows:
        counts = np.array([row['num_keypoints'] for row in stats_rows], dtype=np.float32)
        print('----- 统计摘要 -----')
        print(f"图像数: {len(stats_rows)}")
        print(f"关键点数 min/mean/max: {int(counts.min())} / {counts.mean():.2f} / {int(counts.max())}")
        print(f"stats.csv: {stats_csv_path}")

    print(f'伪标签导出完成: {export_root}')


if __name__ == '__main__':
    main()

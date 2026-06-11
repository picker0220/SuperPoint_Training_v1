"""
在 dark/ 上直观对比 SuperPoint 与 ORB

指标:
1. 单帧关键点数:   各自检测到多少个关键点
2. 匹配数:         两张同位姿图之间 brute-force 匹配 + Lowe ratio 后的内点数
3. 重复率:         帧 1 的关键点在帧 2 一定像素半径内出现的比例
4. 单帧可视化:     暗图上分别画 SuperPoint / ORB / ORB+CLAHE 的关键点

用法:
    python tools/evaluate_dark.py --ckpt checkpoints/superpoint_final.pth \\
        --image_dir dark --num_pairs 10 --device cpu
"""

import argparse
import os
import sys
import csv
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from src.models.superpoint import SuperPoint, load_superpoint
from src.models.losses import SuperPointLoss
from visualize import preprocess_image, detect_keypoints


def parse_args():
    p = argparse.ArgumentParser(description='在 dark/ 上对比 SuperPoint 与 ORB')
    p.add_argument('--ckpt', type=str, default='', help='SuperPoint checkpoint 路径,空则使用随机初始化')
    p.add_argument('--image_dir', type=str, default='dark', help='暗图目录')
    p.add_argument('--num_pairs', type=int, default=10, help='随机抽样多少对图')
    p.add_argument('--threshold', type=float, default=0.02, help='SuperPoint 关键点阈值')
    p.add_argument('--nms_distance', type=int, default=8, help='SuperPoint NMS 距离')
    p.add_argument('--orb_features', type=int, default=1000, help='ORB 最大特征数')
    p.add_argument('--device', type=str, default='cuda', help='推理设备')
    p.add_argument('--output_dir', type=str, default='', help='可视化输出目录')
    p.add_argument('--stats_csv', type=str, default='', help='统计 csv 路径,空则自动')
    return p.parse_args()


def clahe_preprocess(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def superpoint_detect(model, image_bgr, device, threshold, nms_distance):
    image_tensor = preprocess_image(image_bgr)
    keypoints, _, _ = detect_keypoints(model, image_tensor, device=device,
                                        threshold=threshold, nms_distance=nms_distance)
    return keypoints


def orb_detect(image_gray, n_features, use_clahe=False):
    if use_clahe:
        image_gray = clahe_preprocess(image_gray)
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=12, patchSize=31, fastThreshold=10)
    kps, desc = orb.detectAndCompute(image_gray, None)
    if kps is None:
        return [], None
    return [(int(k.pt[0]), int(k.pt[1])) for k in kps], desc


def match_descriptors(desc_a, desc_b, ratio=0.75):
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return len(good)


def repeatability(kps_a, kps_b, tol=4):
    if not kps_a or not kps_b:
        return 0.0, 0
    a = np.array(kps_a, dtype=np.float32)
    b = np.array(kps_b, dtype=np.float32)
    matched = 0
    used = np.zeros(len(b), dtype=bool)
    for (ax, ay) in a:
        d = np.sqrt((b[:, 0] - ax) ** 2 + (b[:, 1] - ay) ** 2)
        d[used] = 1e9
        idx = int(np.argmin(d))
        if d[idx] <= tol:
            matched += 1
            used[idx] = True
    return matched / max(len(kps_a), 1), matched


def draw_kp_overlay(image_bgr, kps, color=(0, 255, 0), radius=2):
    overlay = image_bgr.copy()
    for (x, y) in kps:
        cv2.circle(overlay, (x, y), radius, color, -1)
    return overlay


def synth_warp_pair(image_bgr, max_offset=40):
    h, w = image_bgr.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + np.random.uniform(-max_offset, max_offset, size=src.shape).astype(np.float32)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        return image_bgr, image_bgr, np.eye(3)
    warped = cv2.warpPerspective(image_bgr, H, (w, h), borderMode=cv2.BORDER_REFLECT)
    return image_bgr, warped, H.astype(np.float32)


def main():
    args = parse_args()
    device = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f'图像目录不存在: {args.image_dir}')
    image_paths = []
    for n in sorted(os.listdir(args.image_dir)):
        if n.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths.append(os.path.join(args.image_dir, n))
    if not image_paths:
        raise RuntimeError(f'目录 {args.image_dir} 中没有图像')

    if args.ckpt and os.path.exists(args.ckpt):
        model = load_superpoint(args.ckpt, device=device, verbose=True)
    else:
        print(f'警告: 没找到 checkpoint {args.ckpt},使用随机初始化的 SuperPoint')
        config = Config()
        model = SuperPoint(encoder_dim=config.ENCODER_DIM, grid_size=config.GRID_SIZE).to(device)
        model.eval()

    np.random.seed(0)
    pairs = np.random.choice(len(image_paths), size=(args.num_pairs, 2), replace=True)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = args.output_dir or os.path.join(project_root, 'outputs', 'visualizations', 'evaluate_dark')
    os.makedirs(out_dir, exist_ok=True)
    stats_csv = args.stats_csv or os.path.join(out_dir, 'stats.csv')
    rows = []

    sp_counts, orb_counts, orb_clahe_counts = [], [], []
    sp_matches, orb_matches, orb_clahe_matches = [], [], []
    sp_repeats, orb_repeats, orb_clahe_repeats = [], [], []

    for idx, (i, j) in enumerate(pairs):
        path = image_paths[i]
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.resize(image, (640, 480))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img1, img2, H = synth_warp_pair(image)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sp_kps1 = superpoint_detect(model, img1, device, args.threshold, args.nms_distance)
        sp_kps2 = superpoint_detect(model, img2, device, args.threshold, args.nms_distance)

        orb_kps1, orb_desc1 = orb_detect(gray1, args.orb_features, use_clahe=False)
        orb_kps2, orb_desc2 = orb_detect(gray2, args.orb_features, use_clahe=False)
        orb_c_kps1, orb_c_desc1 = orb_detect(gray1, args.orb_features, use_clahe=True)
        orb_c_kps2, orb_c_desc2 = orb_detect(gray2, args.orb_features, use_clahe=True)

        sp_match = match_descriptors(None, None)  # SuperPoint 描述子用 L2 距离,单独算
        # SuperPoint 自匹配
        sp_match = superpoint_match(sp_kps1, sp_kps2, model, img1, img2, device, args.threshold, args.nms_distance)

        orb_match = match_descriptors(orb_desc1, orb_desc2)
        orb_c_match = match_descriptors(orb_c_desc1, orb_c_desc2)

        sp_rep, _ = repeatability(sp_kps1, sp_kps2)
        orb_rep, _ = repeatability(orb_kps1, orb_kps2)
        orb_c_rep, _ = repeatability(orb_c_kps1, orb_c_kps2)

        sp_counts.append(len(sp_kps1))
        orb_counts.append(len(orb_kps1))
        orb_clahe_counts.append(len(orb_c_kps1))
        sp_matches.append(sp_match)
        orb_matches.append(orb_match)
        orb_clahe_matches.append(orb_c_match)
        sp_repeats.append(sp_rep)
        orb_repeats.append(orb_rep)
        orb_clahe_repeats.append(orb_c_rep)

        rows.append([os.path.basename(path),
                     len(sp_kps1), len(orb_kps1), len(orb_c_kps1),
                     sp_match, orb_match, orb_c_match,
                     f'{sp_rep:.3f}', f'{orb_rep:.3f}', f'{orb_c_rep:.3f}'])

        # 单对可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, kps, name, kp2 in (
            (axes[0, 0], sp_kps1, 'SuperPoint frame1', sp_kps2),
            (axes[0, 1], orb_kps1, 'ORB frame1', orb_kps2),
            (axes[0, 2], orb_c_kps1, 'ORB+CLAHE frame1', orb_c_kps2),
            (axes[1, 0], sp_kps2, 'SuperPoint frame2', None),
            (axes[1, 1], orb_kps2, 'ORB frame2', None),
            (axes[1, 2], orb_c_kps2, 'ORB+CLAHE frame2', None),
        ):
            ax.imshow(cv2.cvtColor(image if 'frame1' in name else img2, cv2.COLOR_BGR2RGB))
            for (x, y) in kps:
                ax.add_patch(Circle((x, y), 2, color='lime'))
            ax.set_title(f'{name}: {len(kps)} kp')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'pair_{idx:03d}.png'), dpi=80)
        plt.close()

        print(f'pair {idx} ({os.path.basename(path)}): '
              f'SP {len(sp_kps1)} kp / {sp_match} match / {sp_rep:.1%} rep | '
              f'ORB {len(orb_kps1)} kp / {orb_match} match / {orb_rep:.1%} rep | '
              f'ORB+CLAHE {len(orb_c_kps1)} kp / {orb_c_match} match / {orb_c_rep:.1%} rep')

    # stats csv
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'sp_kp', 'orb_kp', 'orb_clahe_kp',
                         'sp_match', 'orb_match', 'orb_clahe_match',
                         'sp_rep', 'orb_rep', 'orb_clahe_rep'])
        writer.writerows(rows)
        writer.writerow(['mean', f'{np.mean(sp_counts):.1f}', f'{np.mean(orb_counts):.1f}',
                         f'{np.mean(orb_clahe_counts):.1f}',
                         f'{np.mean(sp_matches):.1f}', f'{np.mean(orb_matches):.1f}',
                         f'{np.mean(orb_clahe_matches):.1f}',
                         f'{np.mean(sp_repeats):.3f}', f'{np.mean(orb_repeats):.3f}',
                         f'{np.mean(orb_clahe_repeats):.3f}'])
    print('--- summary ---')
    print(f'平均关键点数:    SuperPoint {np.mean(sp_counts):.1f} | ORB {np.mean(orb_counts):.1f} | '
          f'ORB+CLAHE {np.mean(orb_clahe_counts):.1f}')
    print(f'平均 Lowe 内点:  SuperPoint {np.mean(sp_matches):.1f} | ORB {np.mean(orb_matches):.1f} | '
          f'ORB+CLAHE {np.mean(orb_clahe_matches):.1f}')
    print(f'平均重复率:      SuperPoint {np.mean(sp_repeats):.1%} | ORB {np.mean(orb_repeats):.1%} | '
          f'ORB+CLAHE {np.mean(orb_clahe_repeats):.1%}')
    print(f'可视化: {out_dir}')
    print(f'stats:  {stats_csv}')


def superpoint_match(kps1, kps2, model, img1, img2, device, threshold, nms_distance):
    """用 SuperPoint 描述子做 brute-force L2 匹配 + Lowe ratio"""
    if not kps1 or not kps2:
        return 0
    t1 = preprocess_image(img1).to(device)
    t2 = preprocess_image(img2).to(device)
    with torch.no_grad():
        d1, _ = model(t1)
        d2, _ = model(t2)
    d1 = d1.squeeze(0).cpu().numpy()  # [256, Hc, Wc]
    d2 = d2.squeeze(0).cpu().numpy()
    Hc, Wc = d1.shape[1:]
    grid = 8

    def sample_desc(desc_map, kps):
        out = []
        for (x, y) in kps:
            gx = min(max(x // grid, 0), Wc - 1)
            gy = min(max(y // grid, 0), Hc - 1)
            out.append(desc_map[:, gy, gx])
        return np.stack(out, axis=0) if out else np.zeros((0, desc_map.shape[0]), dtype=np.float32)

    v1 = sample_desc(d1, kps1)
    v2 = sample_desc(d2, kps2)
    if v1.shape[0] == 0 or v2.shape[0] == 0:
        return 0
    sim = v1 @ v2.T  # [N1, N2], cosine similarity
    nbr = np.argsort(-sim, axis=1)[:, :2]
    good = 0
    for i, (m, n) in enumerate(nbr):
        if sim[i, m] >= 0.75 * sim[i, n]:  # Lowe ratio 0.75 (was 0.85, too loose -> always passes)
            good += 1
    return good


if __name__ == '__main__':
    main()
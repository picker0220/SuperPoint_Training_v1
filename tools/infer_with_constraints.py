"""
单帧推理 + HA 同样的后处理约束

为什么存在: 训出来的 student 会"满天花" / epoch_35 单帧 inference 有 ground blob
(因为模型本身有位置偏置)。HA 的 grid_topk 4 + edge_margin 8 + 50 次 warp 平均
能把这些偏置抹掉,得到干净 label。

这个脚本把那套后处理约束 (grid_topk + edge_margin + NMS) 原样搬到 inference 时用,
等价于"1 次 warp + HA 后处理"的 mini-HA。结果跟 HA label 数量级一致 (~80~120 干净点),
但速度是完整 HA 的 50 倍,够 SLAM 实时用。

用法:
    python tools/infer_with_constraints.py \\
        --ckpt checkpoints/superpoint_epoch_35.pth \\
        --image_dir ~/superpoint/dataset/dark \\
        --output_dir outputs/infer_e35_constraints

    # 想接近 ORB 数量 (~1000),把 grid 分得更细:
    #   --grid_rows 12 --grid_cols 16 --grid_topk 5
    #   12*16*5 = 960 上限,NMS 后 ~700~900
"""

import argparse
import os
import sys
import csv
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.superpoint import load_superpoint
from visualize import preprocess_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/infer_constraints")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--threshold", type=float, default=0.015)
    p.add_argument("--nms_distance", type=int, default=6)
    p.add_argument("--edge_margin", type=int, default=8)
    p.add_argument("--grid_topk", type=int, default=5)
    p.add_argument("--grid_rows", type=int, default=12)
    p.add_argument("--grid_cols", type=int, default=16)
    p.add_argument("--max_images", type=int, default=0, help="最多处理多少张,0=全部")
    p.add_argument("--target_count", type=int, default=0,
                   help="目标 keypoint 数量(超过此值时按分数截断);0=不截断")
    return p.parse_args()


def detect_heatmap(model, image_bgr, device):
    """单帧推理,返回 full-resolution heatmap (H, W) float32"""
    tensor = preprocess_image(image_bgr).to(device)
    with torch.no_grad():
        _, scores = model(tensor)
    prob = torch.softmax(scores, dim=1)[:, :-1, :, :]   # [1, 64, H/8, W/8]
    heatmap = torch.pixel_shuffle(prob, 8)              # [1, 1, H, W]
    return heatmap.squeeze().cpu().numpy()


def select_topk_per_grid(candidates, width, height, grid_rows, grid_cols, grid_topk):
    """跟 HA 里的 select_candidates_with_grid 一模一样,只搬过来不重写"""
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


def detect_keypoints_constrained(heatmap, threshold, nms_distance,
                                 edge_margin, grid_topk, grid_rows, grid_cols,
                                 target_count=0):
    """
    单帧 heatmap -> 受约束的 keypoint 列表
    流程: threshold 收候选 -> edge_margin 削边 -> grid_topk 控分布 -> NMS 去重 -> 可选 target_count 截断
    """
    ys, xs = np.where(heatmap > threshold)
    h, w = heatmap.shape[:2]
    if edge_margin > 0:
        mask = ((ys >= edge_margin) & (ys < h - edge_margin)
                & (xs >= edge_margin) & (xs < w - edge_margin))
        ys, xs = ys[mask], xs[mask]
    candidates = [(float(heatmap[y, x]), int(x), int(y)) for y, x in zip(ys, xs)]
    candidates.sort(reverse=True)
    candidates = select_topk_per_grid(
        candidates, width=w, height=h,
        grid_rows=grid_rows, grid_cols=grid_cols, grid_topk=grid_topk,
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
    if target_count > 0 and len(keypoints) > target_count:
        keypoints = keypoints[:target_count]
    return keypoints


def save_overlay(image_bgr, keypoints, save_path):
    overlay = image_bgr.copy()
    for x, y in keypoints:
        cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)
    cv2.imwrite(save_path, overlay)


def main():
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model = load_superpoint(args.ckpt, device=device, verbose=False)
    model.eval()

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(args.image_dir)
    image_paths = []
    for name in sorted(os.listdir(args.image_dir)):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_paths.append(os.path.join(args.image_dir, name))
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    os.makedirs(args.output_dir, exist_ok=True)
    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    print(f"待处理图像数: {len(image_paths)}")
    print(f"约束参数: threshold={args.threshold} nms={args.nms_distance} "
          f"edge_margin={args.edge_margin} grid_topk={args.grid_topk} "
          f"grid={args.grid_rows}x{args.grid_cols} target_count={args.target_count}")
    print(f"keypoint 上限: {args.grid_rows * args.grid_cols * args.grid_topk}")

    stats = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.resize(image, (640, 480))
        heatmap = detect_heatmap(model, image, device)
        kps = detect_keypoints_constrained(
            heatmap,
            threshold=args.threshold,
            nms_distance=args.nms_distance,
            edge_margin=args.edge_margin,
            grid_topk=args.grid_topk,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            target_count=args.target_count,
        )
        stem = os.path.splitext(os.path.basename(path))[0]
        save_overlay(image, kps, os.path.join(overlay_dir, f"{stem}.png"))
        stats.append([os.path.basename(path), len(kps),
                      float(heatmap.min()), float(heatmap.max()),
                      float(heatmap.mean())])
        print(f"{stem}: {len(kps)} keypoints | heatmap mean={heatmap.mean():.4f}")

    csv_path = os.path.join(args.output_dir, "stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "num_keypoints", "heatmap_min", "heatmap_max", "heatmap_mean"])
        w.writerows(stats)
        if stats:
            counts = [s[1] for s in stats]
            w.writerow(["mean", f"{sum(counts)/len(counts):.1f}",
                        "", "", ""])
    print(f"--- summary ---")
    if stats:
        counts = [s[1] for s in stats]
        print(f"平均 keypoint 数: {sum(counts)/len(counts):.1f} (min {min(counts)} / max {max(counts)})")
    print(f"overlays: {overlay_dir}")
    print(f"stats:    {csv_path}")


if __name__ == "__main__":
    main()

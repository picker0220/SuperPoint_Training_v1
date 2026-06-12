"""
ORB+CLAHE detect + SuperPoint describe hybrid 评估

在 dark 图上对比四种 keypoint+descriptor 组合:
  1. ORB detect + ORB BRIEF (baseline)
  2. ORB+CLAHE detect + ORB BRIEF (CLAHE 增益)
  3. ORB+CLAHE detect + SP describe (hybrid,你这条路)
  4. 纯 SP detect + SP describe (epoch_35 自己的)

关键指标:跨帧 Lowe 内点数。**SP 描述子能否在 ORB 关键点位置上比 ORB BRIEF 匹配得更多**。
如果 (3) > (2),说明 SP 描述子在 dark 上确实比 ORB BRIEF 强,SLAM 跟踪会更稳。

ORB keypoint 位置相同,所以 repeatability 三种方法都一样 (ORB keypoint 的几何重复率);
匹配数差异来自描述子判别力。

用法:
    python tools/orbc_sp_hybrid.py \\
      --sp_ckpt checkpoints/superpoint_epoch_35.pth \\
      --image_dir ~/superpoint/dataset/dark \\
      --num_pairs 20 \\
      --device cuda \\
      --output_dir outputs/orbc_sp_hybrid
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
    p.add_argument("--sp_ckpt", type=str, required=True, help="SuperPoint checkpoint")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--num_pairs", type=int, default=20)
    p.add_argument("--orb_features", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="outputs/orbc_sp_hybrid")
    p.add_argument("--lowe_ratio", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def clahe_preprocess(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def orb_detect_compute(gray, n_features, use_clahe):
    if use_clahe:
        gray = clahe_preprocess(gray)
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=12, patchSize=31, fastThreshold=10)
    kps, desc = orb.detectAndCompute(gray, None)
    if kps is None:
        return [], None
    return [(int(k.pt[0]), int(k.pt[1])) for k in kps], desc


def match_brief(desc_a, desc_b, ratio):
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(desc_a, desc_b, k=2)
    good = 0
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good += 1
    return good


def sp_forward(model, image_bgr, device):
    t = preprocess_image(image_bgr).to(device)
    with torch.no_grad():
        desc, _ = model(t)  # [1, 256, H/8, W/8]
    return desc.squeeze(0).cpu().numpy()


def sample_sp_desc(desc_map, kps, grid=8):
    """在每个 kp 位置采样 SP 描述子 (256-dim)"""
    _, Hc, Wc = desc_map.shape
    out = []
    for (x, y) in kps:
        gx = min(max(x // grid, 0), Wc - 1)
        gy = min(max(y // grid, 0), Hc - 1)
        out.append(desc_map[:, gy, gx])
    return np.stack(out, axis=0) if out else np.zeros((0, desc_map.shape[0]), dtype=np.float32)


def match_sp_desc(v1, v2, ratio):
    """v1, v2: [N, 256] L2 归一化,Lowe ratio on cosine sim"""
    if v1.shape[0] == 0 or v2.shape[0] == 0:
        return 0
    sim = v1 @ v2.T
    nbr = np.argsort(-sim, axis=1)[:, :2]
    good = 0
    for i, (m, n) in enumerate(nbr):
        if sim[i, m] >= ratio * sim[i, n]:
            good += 1
    return good


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


def synth_warp_pair(image_bgr, max_offset=40, seed=None):
    if seed is not None:
        np.random.seed(seed)
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
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    sp_model = load_superpoint(args.sp_ckpt, device=device, verbose=False)
    sp_model.eval()

    image_paths = []
    for n in sorted(os.listdir(args.image_dir)):
        if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_paths.append(os.path.join(args.image_dir, n))
    if not image_paths:
        raise RuntimeError(f"empty image_dir: {args.image_dir}")

    np.random.seed(args.seed)
    pairs = np.random.choice(len(image_paths), size=(args.num_pairs, 2), replace=True)

    os.makedirs(args.output_dir, exist_ok=True)
    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    print(f"device: {device}")
    print(f"图像数: {len(image_paths)}, 配对数: {args.num_pairs}")
    print(f"Lowe ratio: {args.lowe_ratio}")
    print(f"{'pair':>4}  {'ORB':>5}  {'ORB+CL':>6}  {'ORB+CL+SP':>9}  {'SP self':>7}")

    rows = []
    sp_orbclahe, sp_orbbrief, sp_orbc_sp, sp_sp = [], [], [], []
    sp_reps_orbclahe, sp_reps_orbc_sp, sp_reps_sp = [], [], []

    for idx, (i, j) in enumerate(pairs):
        path = image_paths[i]
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.resize(image, (640, 480))
        img1, img2, _ = synth_warp_pair(image, max_offset=40, seed=idx)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 1) ORB baseline (no CLAHE) + ORB BRIEF
        orb_kps1, orb_desc1 = orb_detect_compute(gray1, args.orb_features, use_clahe=False)
        orb_kps2, orb_desc2 = orb_detect_compute(gray2, args.orb_features, use_clahe=False)
        m_orbbrief = match_brief(orb_desc1, orb_desc2, args.lowe_ratio)

        # 2) ORB+CLAHE detect + ORB BRIEF
        orb_c_kps1, orb_c_desc1 = orb_detect_compute(gray1, args.orb_features, use_clahe=True)
        orb_c_kps2, orb_c_desc2 = orb_detect_compute(gray2, args.orb_features, use_clahe=True)
        m_orbc_brief = match_brief(orb_c_desc1, orb_c_desc2, args.lowe_ratio)
        rep_orbclahe, _ = repeatability(orb_c_kps1, orb_c_kps2)

        # 3) ORB+CLAHE detect + SP describe (hybrid)
        desc_map1 = sp_forward(sp_model, img1, device)
        desc_map2 = sp_forward(sp_model, img2, device)
        sp_v1 = sample_sp_desc(desc_map1, orb_c_kps1)
        sp_v2 = sample_sp_desc(desc_map2, orb_c_kps2)
        m_orbc_sp = match_sp_desc(sp_v1, sp_v2, args.lowe_ratio)
        rep_orbc_sp, _ = repeatability(orb_c_kps1, orb_c_kps2)

        # 4) 纯 SP (用 epoch_35 / 你训的 student)
        #    复用 SP forward 拿 keypoint 数量
        #    用 grid_topk 4 控分布(跟 HA 一致)
        #    描述子直接来自 SP forward
        from visualize import detect_keypoints
        sp_kps1, _, _ = detect_keypoints(sp_model, preprocess_image(img1), device,
                                           threshold=0.015, nms_distance=12)
        sp_kps2, _, _ = detect_keypoints(sp_model, preprocess_image(img2), device,
                                           threshold=0.015, nms_distance=12)
        sp_v1_self = sample_sp_desc(desc_map1, sp_kps1)
        sp_v2_self = sample_sp_desc(desc_map2, sp_kps2)
        m_sp_self = match_sp_desc(sp_v1_self, sp_v2_self, args.lowe_ratio)
        rep_sp, _ = repeatability(sp_kps1, sp_kps2)

        sp_orbbrief.append(m_orbbrief)
        sp_orbclahe.append(m_orbc_brief)
        sp_orbc_sp.append(m_orbc_sp)
        sp_sp.append(m_sp_self)
        sp_reps_orbclahe.append(rep_orbclahe)
        sp_reps_orbc_sp.append(rep_orbc_sp)
        sp_reps_sp.append(rep_sp)

        rows.append([os.path.basename(path), len(orb_c_kps1),
                     m_orbbrief, m_orbc_brief, m_orbc_sp, m_sp_self,
                     f"{rep_orbclahe:.3f}", f"{rep_orbc_sp:.3f}", f"{rep_sp:.3f}"])
        print(f"{idx:>4}  {m_orbbrief:>5}  {m_orbc_brief:>6}  {m_orbc_sp:>9}  {m_sp_self:>7}")

    # CSV
    csv_path = os.path.join(args.output_dir, "stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "orb_kp_count",
                    "orb_brief_match", "orb_clahe_brief_match", "orb_clahe_sp_match", "sp_self_match",
                    "rep_orb_clahe", "rep_orb_clahe_sp", "rep_sp_self"])
        w.writerows(rows)
        if rows:
            w.writerow(["mean", f"{np.mean([r[1] for r in rows]):.1f}",
                        f"{np.mean(sp_orbbrief):.1f}", f"{np.mean(sp_orbclahe):.1f}",
                        f"{np.mean(sp_orbc_sp):.1f}", f"{np.mean(sp_sp):.1f}",
                        f"{np.mean(sp_reps_orbclahe):.3f}",
                        f"{np.mean(sp_reps_orbc_sp):.3f}",
                        f"{np.mean(sp_reps_sp):.3f}"])
    print("--- summary ---")
    print(f"avg matches:    ORB-BRIEF={np.mean(sp_orbbrief):.1f}  "
          f"ORB+CL-BRIEF={np.mean(sp_orbclahe):.1f}  "
          f"ORB+CL+SP={np.mean(sp_orbc_sp):.1f}  "
          f"SP self={np.mean(sp_sp):.1f}")
    print(f"avg repeat:     ORB+CL={np.mean(sp_reps_orbclahe):.3f}  "
          f"ORB+CL+SP={np.mean(sp_reps_orbc_sp):.3f}  "
          f"SP self={np.mean(sp_reps_sp):.3f}")
    print(f"stats: {csv_path}")


if __name__ == "__main__":
    main()

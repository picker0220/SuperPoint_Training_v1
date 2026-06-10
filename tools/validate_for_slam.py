"""
SLAM 适用性验证:用 C1 teacher 在 dark/ 上跑 homography 估计,
跟 ORB 比 inlier ratio,看 SuperPoint 特征是否真的对 SLAM 有用。

不需要 ONNX,纯 Python 跑 PyTorch + OpenCV。

用法:
    python tools/validate_for_slam.py \\
        --ckpt checkpoints/superpoint_final.pth \\
        --image_dir ~/superpoint/dataset/dark \\
        --num_pairs 20 \\
        --threshold 0.015 \\
        --output_dir outputs/slam_validate_c1
"""
import argparse
import os
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.superpoint import SuperPoint, load_magicpoint_weights
from visualize import preprocess_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--image_dir', type=str, required=True)
    p.add_argument('--num_pairs', type=int, default=20)
    p.add_argument('--threshold', type=float, default=0.015)
    p.add_argument('--nms_distance', type=int, default=4)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--output_dir', type=str, default='outputs/slam_validate')
    return p.parse_args()


def load_sp_model(ckpt_path, device):
    model = SuperPoint(encoder_dim=256, grid_size=8)
    if 'v6_from_tf' in ckpt_path or 'magicpoint' in ckpt_path:
        model = load_magicpoint_weights(model, ckpt_path, strict=False, verbose=False)
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt['model_state_dict'].items()}
        model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model


def sp_extract(model, image_gray, device, threshold, nms_distance):
    """对单张图跑 SuperPoint,返回 (keypoints Nx2 float32, descriptors NxD float32)"""
    t = preprocess_image(image_gray).to(device)
    with torch.no_grad():
        desc, scores = model(t)
    prob = torch.softmax(scores, dim=1)[:, :-1, :, :]
    heatmap = torch.pixel_shuffle(prob, upscale_factor=8).squeeze().cpu().numpy()
    desc_full = desc.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (Hc, Wc, 256)

    h, w = heatmap.shape
    kps = []
    descs = []
    Hc, Wc = desc_full.shape[:2]
    cell_h = h / Hc
    cell_w = w / Wc
    for y in range(Hc):
        for x in range(Wc):
            v = float(heatmap[y, x])
            if v > threshold:
                # 取该 cell 中心点
                kx = int(x * cell_w + cell_w / 2)
                ky = int(y * cell_h + cell_h / 2)
                kps.append([kx, ky])
                descs.append(desc_full[y, x])
    if not kps:
        return np.zeros((0, 2), np.float32), np.zeros((0, 256), np.float32)
    kps = np.array(kps, np.float32)
    descs = np.array(descs, np.float32)
    # 简易 NMS: 同 8 邻域内只留一个(conf 最高的)
    keep = _simple_nms(kps, heatmap, kps, nms_distance)
    return kps[keep], descs[keep]


def _simple_nms(kps, heatmap, _unused, nms_distance):
    """近似 NMS:用 grid 索引去重,8 邻域只保留 1 个"""
    if len(kps) == 0:
        return np.zeros(0, dtype=bool)
    cell = (kps // nms_distance).astype(np.int32)
    _, idx = np.unique(cell, axis=0, return_index=True)
    return idx


def orb_extract(image_gray, n_features=1000):
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=12, patchSize=31, fastThreshold=10)
    kps, descs = orb.detectAndCompute(image_gray, None)
    if kps is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8)
    pts = np.array([kp.pt for kp in kps], np.float32)
    return pts, descs


def l2_match(desc_a, desc_b, ratio=0.8):
    """Lowe ratio test 匹配,desc_a/b 必须是 L2 归一化的"""
    if len(desc_a) == 0 or len(desc_b) == 0:
        return np.zeros((0, 2), np.int32)
    # cosine similarity = dot product (L2-normed)
    sim = desc_a @ desc_b.T  # (Na, Nb)
    idx_sorted = np.argsort(-sim, axis=1)
    # ratio test
    good = []
    for i in range(sim.shape[0]):
        j1, j2 = idx_sorted[i, 0], idx_sorted[i, 1]
        if sim[i, j1] >= ratio * sim[i, j2]:
            good.append([i, j1])
    return np.array(good, np.int32).reshape(-1, 2) if good else np.zeros((0, 2), np.int32)


def hamming_match(desc_a, desc_b, ratio=0.8):
    """ORB 用的 Hamming 距离 + ratio test"""
    if len(desc_a) == 0 or len(desc_b) == 0:
        return np.zeros((0, 2), np.int32)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append([m.queryIdx, m.trainIdx])
    return np.array(good, np.int32).reshape(-1, 2) if good else np.zeros((0, 2), np.int32)


def evaluate_pair(kps1, kps2, matches, name, tag):
    """估单应 + 数 inlier"""
    if len(matches) < 4:
        return {'name': name, 'tag': tag, 'matches': len(matches), 'inliers': 0, 'inlier_ratio': 0.0, 'H_error': -1}
    pts1 = kps1[matches[:, 0]]
    pts2 = kps2[matches[:, 1]]
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None:
        return {'name': name, 'tag': tag, 'matches': len(matches), 'inliers': 0, 'inlier_ratio': 0.0, 'H_error': -1}
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    inlier_ratio = inliers / len(matches) if len(matches) > 0 else 0.0
    # 算 reprojection error
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    projected = (H @ pts1_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    err = np.linalg.norm(projected - pts2, axis=1)
    H_error = float(err.mean())
    return {
        'name': name, 'tag': tag,
        'matches': len(matches), 'inliers': inliers,
        'inlier_ratio': inlier_ratio, 'H_error': H_error,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    img_dir = os.path.expanduser(args.image_dir)
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
    if len(files) < 2:
        print(f'not enough images in {img_dir}')
        return
    print(f'found {len(files)} images')

    # 取连续对
    import random
    random.seed(0)
    pairs = [(files[i], files[i + 1]) for i in range(len(files) - 1)]
    random.shuffle(pairs)
    pairs = pairs[:args.num_pairs]

    # 加载 SP
    print(f'loading SP model: {args.ckpt}')
    sp_model = load_sp_model(args.ckpt, device)

    rows = []
    for name1, name2 in pairs:
        img1 = cv2.imread(os.path.join(img_dir, name1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(img_dir, name2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue
        img1 = cv2.resize(img1, (640, 480))
        img2 = cv2.resize(img2, (640, 480))

        # SP 提特征 + 匹配
        sp_k1, sp_d1 = sp_extract(sp_model, img1, device, args.threshold, args.nms_distance)
        sp_k2, sp_d2 = sp_extract(sp_model, img2, device, args.threshold, args.nms_distance)
        sp_matches = l2_match(sp_d1, sp_d2, ratio=0.8)
        sp_row = evaluate_pair(sp_k1, sp_k2, sp_matches, f'{name1}+{name2}', 'SP')

        # ORB
        orb_k1, orb_d1 = orb_extract(img1)
        orb_k2, orb_d2 = orb_extract(img2)
        orb_matches = hamming_match(orb_d1, orb_d2, ratio=0.8)
        orb_row = evaluate_pair(orb_k1, orb_k2, orb_matches, f'{name1}+{name2}', 'ORB')

        print(f'{sp_row["name"]}:')
        print(f'  SP:  {sp_row["matches"]:4d} matches, {sp_row["inliers"]:3d} inliers, ratio={sp_row["inlier_ratio"]:.2%}, H_err={sp_row["H_error"]:.2f}px')
        print(f'  ORB: {orb_row["matches"]:4d} matches, {orb_row["inliers"]:3d} inliers, ratio={orb_row["inlier_ratio"]:.2%}, H_err={orb_row["H_error"]:.2f}px')

        rows.append(sp_row)
        rows.append(orb_row)

    # 写 CSV
    csv_path = os.path.join(args.output_dir, 'slam_validate.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=['name', 'tag', 'matches', 'inliers', 'inlier_ratio', 'H_error'])
        writer.writeheader()
        writer.writerows(rows)

    # 汇总
    sp_rows = [r for r in rows if r['tag'] == 'SP']
    orb_rows = [r for r in rows if r['tag'] == 'ORB']
    if sp_rows and orb_rows:
        sp_inlier_ratio = np.mean([r['inlier_ratio'] for r in sp_rows])
        orb_inlier_ratio = np.mean([r['inlier_ratio'] for r in orb_rows])
        sp_matches_mean = np.mean([r['matches'] for r in sp_rows])
        orb_matches_mean = np.mean([r['matches'] for r in orb_rows])
        sp_inliers_mean = np.mean([r['inliers'] for r in sp_rows])
        orb_inliers_mean = np.mean([r['inliers'] for r in orb_rows])
        sp_H_error = np.mean([r['H_error'] for r in sp_rows if r['H_error'] > 0])
        orb_H_error = np.mean([r['H_error'] for r in orb_rows if r['H_error'] > 0])

        print()
        print('=' * 60)
        print('SLAM 适用性验证汇总')
        print('=' * 60)
        print(f'{"":20s}  {"SP":>15s}  {"ORB":>15s}')
        print(f'{"平均匹配数":20s}  {sp_matches_mean:>15.1f}  {orb_matches_mean:>15.1f}')
        print(f'{"平均 inliers":20s}  {sp_inliers_mean:>15.1f}  {orb_inliers_mean:>15.1f}')
        print(f'{"平均 inlier ratio":20s}  {sp_inlier_ratio:>15.2%}  {orb_inlier_ratio:>15.2%}')
        if sp_H_error > 0 and orb_H_error > 0:
            print(f'{"平均 H 误差 (px)":20s}  {sp_H_error:>15.2f}  {orb_H_error:>15.2f}')
        print()
        print(f'CSV: {csv_path}')
        if sp_inlier_ratio > orb_inlier_ratio:
            print(f'>>> SP 赢了: inlier ratio {sp_inlier_ratio:.2%} > ORB {orb_inlier_ratio:.2%}')
        else:
            print(f'>>> ORB 赢了: inlier ratio {orb_inlier_ratio:.2%} > SP {sp_inlier_ratio:.2%}')


if __name__ == '__main__':
    main()
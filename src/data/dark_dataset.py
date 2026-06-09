"""
Dark Image Dataset

从真实夜间/低光图像 + 已生成的同位姿伪标签训练 detector / descriptor。

预期目录结构:
    dark/                   # 真实图像
        IMG_1972.JPG
        IMG_1980.JPG
        ...
    outputs/pseudo_labels/<export_name>/keypoints/
        IMG_1972.npy        # 关键点坐标 (N, 2), int
        ...

训练时,对每张图:
1. 加载原图与伪标签
2. 在线做随机单应变换,得到 image2 + valid correspondence mask
3. 在线做光度增强(gamma/噪声/对比度),image1 / image2 共享同一增强策略,
   保证描述子学习的是"同一物理点的不同视角",而不是"两张不同亮度的图"。

返回统一接口 (image1, image2, labels, valid_mask)。
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .lowlight_synthetic_dataset import (
    sample_homography,
    warp_image,
    compute_valid_mask,
    compute_warp_grid_from_homography,
    LowLightAugmentor,
)


def _build_65class_labels(keypoints, h, w, grid_size=8):
    """将关键点 (N,2) 编码为标准 65 类标签 (H/8, W/8)"""
    labels = np.full((h // grid_size, w // grid_size), 64, dtype=np.int64)
    for (x, y) in keypoints:
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        gx, gy = x // grid_size, y // grid_size
        if 0 <= gx < labels.shape[1] and 0 <= gy < labels.shape[0]:
            cx = x % grid_size
            cy = y % grid_size
            labels[gy, gx] = cy * grid_size + cx
    return labels


class DarkImageDataset(Dataset):
    def __init__(self, image_dir, keypoint_dir=None,
                 image_height=480, image_width=640, grid_size=8,
                 use_lowlight_aug=True, max_warp_offset=24,
                 fallback_min_corners=20):
        self.image_dir = image_dir
        self.keypoint_dir = keypoint_dir
        self.image_height = image_height
        self.image_width = image_width
        self.grid_size = grid_size
        self.use_lowlight_aug = use_lowlight_aug
        self.max_warp_offset = max_warp_offset
        self.fallback_min_corners = fallback_min_corners
        self.augmentor = LowLightAugmentor() if use_lowlight_aug else None
        self.samples = self._collect()

    def _collect(self):
        if not os.path.isdir(self.image_dir):
            return []
        out = []
        for name in sorted(os.listdir(self.image_dir)):
            if not name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            stem = os.path.splitext(name)[0]
            img_path = os.path.join(self.image_dir, name)
            kp_path = None
            if self.keypoint_dir:
                cand = os.path.join(self.keypoint_dir, f'{stem}.npy')
                if os.path.exists(cand):
                    kp_path = cand
            out.append((img_path, kp_path))
        return out

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        if img.shape[:2] != (self.image_height, self.image_width):
            img = cv2.resize(img, (self.image_width, self.image_height))
        return img

    def _load_keypoints(self, path):
        if not path or not os.path.exists(path):
            return None
        kp = np.load(path)
        if kp.ndim == 2 and kp.shape[1] == 2:
            return [(int(p[0]), int(p[1])) for p in kp]
        return None

    def _fallback_keypoints(self, image):
        """若没有伪标签,用 Shi-Tomasi 兜底"""
        g = np.float32(image) / 255.0
        corners = cv2.goodFeaturesToTrack(
            g, maxCorners=self.fallback_min_corners * 5,
            qualityLevel=0.005, minDistance=8,
        )
        if corners is None:
            return []
        return [(int(c[0][0]), int(c[0][1])) for c in corners]

    def __getitem__(self, idx):
        img_path, kp_path = self.samples[idx]
        image = self._load_image(img_path)
        keypoints = self._load_keypoints(kp_path)
        if not keypoints:
            keypoints = self._fallback_keypoints(image)

        # 光度增强(只对 image1 增强,然后 warp)
        if self.use_lowlight_aug:
            image1 = self.augmentor(image)
        else:
            image1 = image.copy()

        # 单应 warp -> image2
        H, _ = sample_homography(self.image_height, self.image_width,
                                  max_offset=self.max_warp_offset)
        image2 = warp_image(image1, H, self.image_height, self.image_width)

        labels = _build_65class_labels(keypoints, self.image_height, self.image_width,
                                       grid_size=self.grid_size)
        valid_mask = compute_valid_mask(keypoints, H, self.image_height, self.image_width,
                                         grid_size=self.grid_size)

        image1_t = torch.from_numpy(image1).float().unsqueeze(0) / 255.0
        image2_t = torch.from_numpy(image2).float().unsqueeze(0) / 255.0
        labels_t = torch.from_numpy(labels).long()
        valid_mask_t = torch.from_numpy(valid_mask).bool()
        hc = self.image_height // self.grid_size
        wc = self.image_width // self.grid_size
        warp_grid = compute_warp_grid_from_homography(H, hc, wc, grid_size=self.grid_size)
        return image1_t, image2_t, labels_t, valid_mask_t, warp_grid


def get_dark_dataloader(image_dir, keypoint_dir=None, batch_size=4, num_workers=0,
                        image_height=480, image_width=640, grid_size=8,
                        use_lowlight_aug=True, max_warp_offset=24):
    dataset = DarkImageDataset(
        image_dir=image_dir,
        keypoint_dir=keypoint_dir,
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        use_lowlight_aug=use_lowlight_aug,
        max_warp_offset=max_warp_offset,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )



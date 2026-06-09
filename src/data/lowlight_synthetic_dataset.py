"""
Low-Light Synthetic Shapes Dataset

基于 synthetic_shapes_dataset.py 的几何形状生成,叠加真实低光物理模型,
用于让 detector 在大量"暗的、噪声大的"图像上学到"什么在暗处依然稳定"。

低光物理模型组成:
1. Gamma 校正: 模拟相机响应曲线与曝光不足,gamma ~ U(0.2, 0.7)
2. Shot noise:  泊松分布,模拟光子计数噪声(暗部相对更明显)
3. Read noise:  高斯分布,模拟传感器读出噪声
4. 对比度衰减: 暗部被压向黑色,亮部被截断
5. 局部亮斑:   随机加入 1-3 个高斯亮斑,模拟路灯/窗户/招牌
6. 轻模糊:     模拟长曝光下相机的微抖与离焦

输出统一接口 (image1, image2, labels, valid_mask),其中:
- image1: [1, H, W]   低光物理模型后的图像
- image2: [1, H, W]   image1 经随机单应变换后的图像,用于描述子监督
- labels: [H/8, W/8]  65 类 SuperPoint 检测标签 (基于 image1 的关键点)
- valid_mask: [H/8, W/8] bool  标识 image1 中哪些 cell 在 image2 中有有效对应
"""

import math
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .synthetic_shapes_dataset import SyntheticShapesDataset




def compute_warp_grid_from_homography(H, hc, wc, grid_size=8, device=None):
    """
    给定 3x3 单应矩阵 H 与描述子网格大小 (hc, wc),
    计算 image1 的每个 cell 中心映射到 image2 cell 空间的归一化坐标。

    返回 [hc, wc, 2] 张量,可直接喂给 F.grid_sample。
    """
    ys = (torch.arange(hc, dtype=torch.float32) * grid_size + grid_size / 2.0)
    xs = (torch.arange(wc, dtype=torch.float32) * grid_size + grid_size / 2.0)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    ones = torch.ones_like(xx)
    pts = torch.stack([xx, yy, ones], dim=-1).reshape(-1, 3)  # [N, 3]
    H_t = torch.as_tensor(H, dtype=torch.float32)
    warped = pts @ H_t.T
    warped = warped[:, :2] / (warped[:, 2:3] + 1e-8)
    warped_cells = warped / float(grid_size)
    x_n = warped_cells[:, 0] / max(wc - 1, 1) * 2.0 - 1.0
    y_n = warped_cells[:, 1] / max(hc - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x_n, y_n], dim=-1).reshape(hc, wc, 2)
    if device is not None:
        grid = grid.to(device)
    return grid


def identity_warp_grid(hc, wc, device=None):
    g = compute_warp_grid_from_homography(np.eye(3, dtype=np.float32), hc, wc, device=device)
    return g
class LowLightAugmentor:
    """物理模型驱动的低光增强器,作用于 [0, 255] uint8 灰度图"""

    def __init__(self, p_lowlight=0.85, p_normal=0.15):
        # 整体偏向低光,但保留少量正常光样本,避免灾难性遗忘
        self.p_lowlight = p_lowlight
        self.p_normal = p_normal

    def __call__(self, image):
        if random.random() < self.p_lowlight:
            return self._lowlight_pipeline(image)
        return self._normal_pipeline(image)

    def _normal_pipeline(self, image):
        img = image.astype(np.float32)
        # 轻度 gamma / 噪声 / 模糊
        if random.random() < 0.6:
            gamma = random.uniform(0.8, 1.2)
            img = np.power(img / 255.0, gamma) * 255.0
        if random.random() < 0.5:
            noise = np.random.normal(0, random.uniform(2, 6), img.shape)
            img += noise
        if random.random() < 0.3:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _lowlight_pipeline(self, image):
        img = image.astype(np.float32)

        # 1) 强 gamma 压暗,模拟曝光不足
        gamma = random.uniform(0.20, 0.70)
        img = np.power(img / 255.0, gamma) * 255.0

        # 2) Shot noise (泊松),暗部相对更明显
        if random.random() < 0.85:
            scale = random.uniform(8.0, 25.0)
            scaled = np.maximum(img, 0) / 255.0 * scale
            noisy = np.random.poisson(scaled).astype(np.float32) / scale * 255.0
            img = noisy

        # 3) Read noise (高斯)
        read_sigma = random.uniform(1.5, 6.0)
        img += np.random.normal(0, read_sigma, img.shape)

        # 4) 对比度衰减,暗部压向 0
        if random.random() < 0.7:
            black_point = random.uniform(0, 30)
            img = np.clip(img - black_point, 0, 255)

        # 5) 局部高斯亮斑,模拟路灯/窗户
        n_spots = random.randint(1, 3)
        for _ in range(n_spots):
            cx = random.randint(0, image.shape[1] - 1)
            cy = random.randint(0, image.shape[0] - 1)
            radius = random.randint(20, 90)
            intensity = random.uniform(120, 255)
            self._add_light_spot(img, cx, cy, radius, intensity)

        # 6) 轻模糊
        if random.random() < 0.5:
            k = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), 0)

        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _add_light_spot(canvas, cx, cy, radius, intensity):
        h, w = canvas.shape
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        falloff = np.exp(-(dist ** 2) / (2 * (radius / 2.0) ** 2 + 1e-3))
        canvas += (falloff * intensity).astype(np.float32)


def sample_homography(h, w, max_perspective=0.04, max_offset=24):
    """
    在合理范围采样一个单应矩阵,模拟手持相机的轻微视角变化。

    Returns:
        H: 3x3 forward warp 矩阵 (src -> dst)
        H_inv: 3x3 inverse
    """
    src = np.float32([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ]).reshape(-1, 1, 2)
    dst = src.copy()
    # 角点扰动
    jitter = np.random.uniform(-max_offset, max_offset, size=src.shape).astype(np.float32)
    dst = dst + jitter
    # 轻微 perspective
    if random.random() < 0.5:
        dst = dst + np.random.uniform(-max_perspective * w, max_perspective * w, size=src.shape)

    H, _ = cv2.findHomography(src, dst)
    if H is None:
        H = np.eye(3, dtype=np.float32)
    H_inv = np.linalg.inv(H)
    return H.astype(np.float32), H_inv.astype(np.float32)


def warp_image(image, H, h, w):
    return cv2.warpPerspective(image, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def compute_valid_mask(keypoints, H, h, w, grid_size=8):
    """
    对每个关键点计算其在 image2 中的对应位置是否在边界内,
    落到对应的 H/8 x W/8 cell。
    """
    mask = np.zeros((h // grid_size, w // grid_size), dtype=bool)
    if len(keypoints) == 0:
        return mask
    pts = np.array(keypoints, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    for (x, y) in warped:
        if 0 <= x < w and 0 <= y < h:
            gx = int(x) // grid_size
            gy = int(y) // grid_size
            if 0 <= gx < mask.shape[1] and 0 <= gy < mask.shape[0]:
                mask[gy, gx] = True
    return mask


class LowLightSyntheticDataset(Dataset):
    """
    包装 synthetic_shapes 生成器,在其上加低光物理模型,并产出图像对用于描述子监督。
    """

    def __init__(self, num_samples=20000, image_height=480, image_width=640, grid_size=8,
                 use_lowlight=True, max_warp_offset=24):
        self.num_samples = num_samples
        self.image_height = image_height
        self.image_width = image_width
        self.grid_size = grid_size
        self.max_warp_offset = max_warp_offset
        self.use_lowlight = use_lowlight

        # 复用现有形状生成器
        self._shape = SyntheticShapesDataset(
            num_samples=num_samples,
            image_height=image_height,
            image_width=image_width,
            grid_size=grid_size,
        )
        self.augmentor = LowLightAugmentor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) 生成形状图
        canvas, keypoints = self._shape._generate_shapes_image()
        keypoints = self._shape._deduplicate_keypoints(keypoints)

        # 2) 应用低光物理模型
        if self.use_lowlight:
            image1 = self.augmentor(canvas)
        else:
            image1 = self.augmentor._normal_pipeline(canvas)

        # 3) 采样单应并 warp 得到 image2
        H, _ = sample_homography(self.image_height, self.image_width,
                                  max_offset=self.max_warp_offset)
        image2 = warp_image(image1, H, self.image_height, self.image_width)

        # 4) 生成 65 类标签 (基于原图关键点)
        labels = self._shape._create_labels(keypoints)
        valid_mask = compute_valid_mask(keypoints, H, self.image_height, self.image_width,
                                         grid_size=self.grid_size)

        # 5) 转 tensor
        image1_t = torch.from_numpy(image1).float().unsqueeze(0) / 255.0
        image2_t = torch.from_numpy(image2).float().unsqueeze(0) / 255.0
        labels_t = torch.from_numpy(labels).long()
        valid_mask_t = torch.from_numpy(valid_mask).bool()
        warp_grid = compute_warp_grid_from_homography(H, self.image_height // self.grid_size,
                                                    self.image_width // self.grid_size,
                                                    grid_size=self.grid_size)
        return image1_t, image2_t, labels_t, valid_mask_t, warp_grid


def get_lowlight_synthetic_dataloader(batch_size=8, num_workers=0, split='train',
                                      image_height=480, image_width=640, grid_size=8,
                                      use_lowlight=True):
    if split == 'train':
        num_samples = 20000
    else:
        num_samples = 2000

    dataset = LowLightSyntheticDataset(
        num_samples=num_samples,
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        use_lowlight=use_lowlight,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )


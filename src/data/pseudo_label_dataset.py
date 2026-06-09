"""
Pseudo Label 数据集

读取真实图像与对应的伪标签关键点坐标,并编码成标准 65 类标签。
当前版本额外支持:
1. 加载聚合热图作为 soft supervision
2. 对输入图像施加低光照 / 低纹理退化增强,逼近 SLAM 前端真实场景
3. 为 student 训练提供第二个光照扰动视图,用于结构一致性约束

C1 流程新增(P1 补丁):
4. enable_night_preprocess: 输入图像先做 CLAHE + gamma 提亮(与 HA 工具的
   apply_night_preprocess 行为一致),让网络能"看见"暗处结构
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PseudoLabelDataset(Dataset):
    def __init__(self, image_dir, keypoint_dir, heatmap_dir='', image_height=480, image_width=640,
                 grid_size=8, enable_low_light_aug=False, enable_night_preprocess=False,
                 clahe_clip_limit=2.5, clahe_grid_size=8, gamma_lift=0.8, enable_denoise=False):
        self.image_dir = image_dir
        self.keypoint_dir = keypoint_dir
        self.heatmap_dir = heatmap_dir
        self.image_height = image_height
        self.image_width = image_width
        self.grid_size = grid_size
        self.enable_low_light_aug = enable_low_light_aug
        self.enable_night_preprocess = enable_night_preprocess
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.gamma_lift = gamma_lift
        self.enable_denoise = enable_denoise
        self.samples = []
        self._collect()

    def _collect(self):
        if not os.path.exists(self.image_dir) or not os.path.exists(self.keypoint_dir):
            return

        for name in sorted(os.listdir(self.image_dir)):
            if not name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            stem = os.path.splitext(name)[0]
            kp_path = os.path.join(self.keypoint_dir, f'{stem}.npy')
            heatmap_path = os.path.join(self.heatmap_dir, f'{stem}.npy') if self.heatmap_dir else ''
            if os.path.exists(kp_path):
                self.samples.append((os.path.join(self.image_dir, name), kp_path, heatmap_path))

    def __len__(self):
        return len(self.samples)

    def _apply_night_preprocess(self, image):
        """暗图 CLAHE 提亮预处理 (与 tools/generate_homographic_labels.py apply_night_preprocess 一致)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.enable_denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
        gray_float = gray.astype(np.float32) / 255.0
        gray_float = np.power(np.clip(gray_float, 0.0, 1.0), self.gamma_lift)
        gray_lifted = np.clip(gray_float * 255.0, 0.0, 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=max(0.1, float(self.clahe_clip_limit)),
            tileGridSize=(max(1, int(self.clahe_grid_size)), max(1, int(self.clahe_grid_size))),
        )
        enhanced = clahe.apply(gray_lifted)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _create_labels(self, keypoints):
        labels = np.full((self.image_height // self.grid_size, self.image_width // self.grid_size), 64, dtype=np.int64)
        for x, y in keypoints:
            x = int(np.clip(x, 0, self.image_width - 1))
            y = int(np.clip(y, 0, self.image_height - 1))
            gx = x // self.grid_size
            gy = y // self.grid_size
            if 0 <= gx < labels.shape[1] and 0 <= gy < labels.shape[0]:
                cell_x = x % self.grid_size
                cell_y = y % self.grid_size
                labels[gy, gx] = cell_y * self.grid_size + cell_x
        return labels

    def _apply_low_light_augmentation(self, image):
        image = image.astype(np.float32) / 255.0

        gamma = np.random.uniform(1.5, 3.0)
        image = np.power(np.clip(image, 0.0, 1.0), gamma)

        brightness_scale = np.random.uniform(0.45, 0.85)
        contrast_scale = np.random.uniform(0.70, 1.05)
        image = np.clip((image - 0.5) * contrast_scale + 0.5, 0.0, 1.0)
        image = np.clip(image * brightness_scale, 0.0, 1.0)

        if np.random.rand() < 0.7:
            noise_std = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        if np.random.rand() < 0.4:
            ksize = int(np.random.choice([3, 5]))
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=np.random.uniform(0.5, 1.5))

        if np.random.rand() < 0.3:
            shadow = np.ones((self.image_height, self.image_width, 1), dtype=np.float32)
            x0 = np.random.randint(0, self.image_width // 2)
            y0 = np.random.randint(0, self.image_height // 2)
            x1 = np.random.randint(self.image_width // 2, self.image_width)
            y1 = np.random.randint(self.image_height // 2, self.image_height)
            shadow[y0:y1, x0:x1] *= np.random.uniform(0.4, 0.8)
            image = np.clip(image * shadow, 0.0, 1.0)

        return (image * 255.0).astype(np.uint8)

    def _apply_consistency_augmentation(self, image):
        image = image.astype(np.float32) / 255.0

        gamma = np.random.uniform(0.9, 1.4)
        image = np.power(np.clip(image, 0.0, 1.0), gamma)

        brightness_scale = np.random.uniform(0.85, 1.10)
        contrast_scale = np.random.uniform(0.90, 1.10)
        image = np.clip((image - 0.5) * contrast_scale + 0.5, 0.0, 1.0)
        image = np.clip(image * brightness_scale, 0.0, 1.0)

        if np.random.rand() < 0.5:
            noise_std = np.random.uniform(0.005, 0.03)
            noise = np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        if np.random.rand() < 0.25:
            ksize = int(np.random.choice([3, 5]))
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=np.random.uniform(0.4, 1.0))

        return (image * 255.0).astype(np.uint8)

    def _load_soft_heatmap(self, heatmap_path):
        if not heatmap_path or not os.path.exists(heatmap_path):
            return np.zeros((self.image_height, self.image_width), dtype=np.float32)

        heatmap = np.load(heatmap_path).astype(np.float32)
        if heatmap.shape != (self.image_height, self.image_width):
            heatmap = cv2.resize(heatmap, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)

        heatmap = np.maximum(heatmap, 0.0)
        max_val = float(heatmap.max())
        if max_val > 1e-6:
            heatmap = heatmap / max_val
        return heatmap.astype(np.float32)

    def __getitem__(self, idx):
        image_path, keypoint_path, heatmap_path = self.samples[idx]
        image = cv2.imread(image_path)
        if image is None:
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        else:
            image = cv2.resize(image, (self.image_width, self.image_height))

        if self.enable_night_preprocess:
            image = self._apply_night_preprocess(image)

        if self.enable_low_light_aug:
            image_main = self._apply_low_light_augmentation(image)
            image_consistency = self._apply_consistency_augmentation(image_main)
        else:
            image_main = image.copy()
            image_consistency = self._apply_consistency_augmentation(image)

        keypoints = np.load(keypoint_path)
        labels = self._create_labels(keypoints)
        soft_heatmap = self._load_soft_heatmap(heatmap_path)

        main_tensor = torch.from_numpy(image_main).float().permute(2, 0, 1) / 255.0
        consistency_tensor = torch.from_numpy(image_consistency).float().permute(2, 0, 1) / 255.0
        image_gray = main_tensor.mean(dim=0, keepdim=True)
        image_consistency_gray = consistency_tensor.mean(dim=0, keepdim=True)
        soft_heatmap_tensor = torch.from_numpy(soft_heatmap).float().unsqueeze(0)
        label_tensor = torch.from_numpy(labels).long()

        return image_gray, image_consistency_gray, soft_heatmap_tensor, label_tensor


def get_pseudo_label_dataloader(image_dir, keypoint_dir, batch_size=4, num_workers=0,
                                image_height=480, image_width=640, grid_size=8,
                                heatmap_dir='', enable_low_light_aug=False,
                                enable_night_preprocess=False,
                                clahe_clip_limit=2.5, clahe_grid_size=8,
                                gamma_lift=0.8, enable_denoise=False):
    dataset = PseudoLabelDataset(
        image_dir=image_dir,
        keypoint_dir=keypoint_dir,
        heatmap_dir=heatmap_dir,
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
        enable_low_light_aug=enable_low_light_aug,
        enable_night_preprocess=enable_night_preprocess,
        clahe_clip_limit=clahe_clip_limit,
        clahe_grid_size=clahe_grid_size,
        gamma_lift=gamma_lift,
        enable_denoise=enable_denoise,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
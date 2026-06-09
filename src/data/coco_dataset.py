"""
COCO数据集加载器

当前版本输出标准 65 类 SuperPoint 检测标签：
- 0~63: 8x8 cell 内子位置
- 64: dustbin，无关键点

说明：
1. 训练入口当前主要使用检测标签，不训练描述子损失
2. 第二张图像仍保留，便于后续继续接入单应性约束的描述子训练
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, root_dir, split='train', image_height=480, image_width=640, use_synthetic_pairs=True):
        self.root_dir = root_dir
        self.split = split
        self.image_height = image_height
        self.image_width = image_width
        self.use_synthetic_pairs = use_synthetic_pairs

        self.image_dir = os.path.join(root_dir, f'{split}2017')
        self.image_paths = []
        self._collect_images()

        print(f"COCO {'训练' if split == 'train' else '验证'}集: {len(self.image_paths)} 张图像")

    def _collect_images(self):
        if not os.path.exists(self.image_dir):
            return

        for filename in sorted(os.listdir(self.image_dir)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(self.image_dir, filename))

    def _extract_keypoints(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = np.float32(gray) / 255.0
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=10
        )

        if corners is None:
            return []

        keypoints = []
        for corner in corners:
            x, y = corner[0]
            keypoints.append((int(x), int(y)))

        return keypoints

    def _create_labels(self, keypoints, height, width, grid_size=8):
        labels = np.full((height // grid_size, width // grid_size), 64, dtype=np.int64)

        for (x, y) in keypoints:
            if not (0 <= x < width and 0 <= y < height):
                continue
            gx = x // grid_size
            gy = y // grid_size
            if 0 <= gx < labels.shape[1] and 0 <= gy < labels.shape[0]:
                cell_x = x % grid_size
                cell_y = y % grid_size
                cls = cell_y * grid_size + cell_x
                labels[gy, gx] = cls

        return labels

    def _apply_random_homography(self, image):
        h, w = image.shape[:2]

        src_points = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])

        max_offset = 50
        dst_points = src_points.copy()
        for i in range(4):
            dst_points[i] += np.random.uniform(-max_offset, max_offset, 2)

        H, _ = cv2.findHomography(src_points, dst_points)
        warped = cv2.warpPerspective(image, H, (w, h), borderMode=cv2.BORDER_REFLECT)
        return warped

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        if image is None:
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        else:
            image = cv2.resize(image, (self.image_width, self.image_height))

        if self.use_synthetic_pairs:
            image2 = self._apply_random_homography(image)
        else:
            image2 = image.copy()

        keypoints = self._extract_keypoints(image)
        labels = self._create_labels(keypoints, self.image_height, self.image_width)

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        image2_tensor = torch.from_numpy(image2).float().permute(2, 0, 1) / 255.0

        image_gray = image_tensor.mean(dim=0, keepdim=True)
        image2_gray = image2_tensor.mean(dim=0, keepdim=True)
        label_tensor = torch.from_numpy(labels).long()

        return image_gray, image2_gray, label_tensor


class COCOPretrainDataset(Dataset):
    def __init__(self, root_dir, split='train', image_height=480, image_width=640):
        self.root_dir = root_dir
        self.split = split
        self.image_height = image_height
        self.image_width = image_width

        self.image_dir = os.path.join(root_dir, f'{split}2017')
        self.image_paths = []
        self._collect_images()

        print(f"COCO预训练 {'训练' if split == 'train' else '验证'}集: {len(self.image_paths)} 张图像")

    def _collect_images(self):
        if not os.path.exists(self.image_dir):
            return

        for filename in sorted(os.listdir(self.image_dir)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(self.image_dir, filename))

    def _extract_keypoints(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = np.float32(gray) / 255.0
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=10
        )

        if corners is None or len(corners) < 10:
            return []

        return [(int(c[0][0]), int(c[0][1])) for c in corners]

    def _create_labels(self, keypoints, height, width, grid_size=8):
        labels = np.full((height // grid_size, width // grid_size), 64, dtype=np.int64)
        for (x, y) in keypoints:
            if not (0 <= x < width and 0 <= y < height):
                continue
            gx, gy = x // grid_size, y // grid_size
            if 0 <= gx < labels.shape[1] and 0 <= gy < labels.shape[0]:
                cell_x = x % grid_size
                cell_y = y % grid_size
                labels[gy, gx] = cell_y * grid_size + cell_x
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        if image is None:
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        else:
            image = cv2.resize(image, (self.image_width, self.image_height))

        keypoints = self._extract_keypoints(image)
        labels = self._create_labels(keypoints, self.image_height, self.image_width)

        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        image_gray = image_tensor.mean(dim=0, keepdim=True)
        label_tensor = torch.from_numpy(labels).long()

        return image_gray, image_gray, label_tensor


def get_coco_dataloader(root_dir, batch_size=4, num_workers=4,
                        split='train', image_height=480, image_width=640,
                        use_synthetic_pairs=True):
    from torch.utils.data import DataLoader

    dataset = COCODataset(
        root_dir=root_dir,
        split=split,
        image_height=image_height,
        image_width=image_width,
        use_synthetic_pairs=use_synthetic_pairs
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return loader
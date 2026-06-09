"""
HPatches数据集加载器

用于加载HPatches数据集，生成SuperPoint训练所需的图像对与标签。
当前版本输出标准 65 类 SuperPoint 检测标签：
- 0~63: 8x8 cell 内子位置
- 64: dustbin，无关键点
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HPatchesDataset(Dataset):
    """
    HPatches监督数据集
    使用传统角点作为伪标签，生成标准 65 类 SuperPoint 检测标签
    """

    def __init__(self, root_dir, split='train', image_height=480, image_width=640, use_hard=True):
        self.root_dir = root_dir
        self.split = split
        self.image_height = image_height
        self.image_width = image_width
        self.use_hard = use_hard

        if os.path.exists(os.path.join(root_dir, 'hpatches', 'hpatches-release')):
            self.dataset_dir = os.path.join(root_dir, 'hpatches', 'hpatches-release')
        elif os.path.exists(os.path.join(root_dir, 'hpatches-release')):
            self.dataset_dir = os.path.join(root_dir, 'hpatches-release')
        else:
            self.dataset_dir = root_dir

        self.sequences = []
        self._collect_sequences()

        np.random.seed(42)
        n_total = len(self.sequences)
        n_train = int(n_total * 0.8)
        if split == 'train':
            self.sequences = self.sequences[:n_train]
        else:
            self.sequences = self.sequences[n_train:]

        print(f"HPatches {'训练' if split == 'train' else '验证'}集: {len(self.sequences)} 个序列")

    def _collect_sequences(self):
        if not os.path.exists(self.dataset_dir):
            return

        for seq_name in sorted(os.listdir(self.dataset_dir)):
            seq_path = os.path.join(self.dataset_dir, seq_name)
            if os.path.isdir(seq_path):
                ref_path = os.path.join(seq_path, 'ref.png')
                if os.path.exists(ref_path):
                    self.sequences.append({
                        'name': seq_name,
                        'path': seq_path,
                        'type': 'illumination' if seq_name.startswith('i_') else 'viewpoint'
                    })

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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_path = seq['path']

        ref_img = cv2.imread(os.path.join(seq_path, 'ref.png'))
        if ref_img is None:
            ref_img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        ref_img = cv2.resize(ref_img, (self.image_width, self.image_height))

        img_index = np.random.randint(1, 6)
        transform_type = 'h' if self.use_hard else 'e'
        warp_img_path = os.path.join(seq_path, f'{transform_type}{img_index}.png')

        if os.path.exists(warp_img_path):
            warp_img = cv2.imread(warp_img_path)
            warp_img = cv2.resize(warp_img, (self.image_width, self.image_height))
        else:
            warp_img = ref_img.copy()

        keypoints = self._extract_keypoints(ref_img)
        if len(keypoints) < 5:
            keypoints = []
            step = 40
            for y in range(step, self.image_height - step, step):
                for x in range(step, self.image_width - step, step):
                    keypoints.append((x, y))

        labels = self._create_labels(keypoints, self.image_height, self.image_width)

        ref_tensor = torch.from_numpy(ref_img).float().permute(2, 0, 1) / 255.0
        warp_tensor = torch.from_numpy(warp_img).float().permute(2, 0, 1) / 255.0

        ref_gray = ref_tensor.mean(dim=0, keepdim=True)
        warp_gray = warp_tensor.mean(dim=0, keepdim=True)
        label_tensor = torch.from_numpy(labels).long()

        return ref_gray, warp_gray, label_tensor


class HPatchesDatasetSSL(Dataset):
    """
    兼容原训练入口的数据集类
    当前同样返回标准 65 类检测标签
    """

    def __init__(self, root_dir, split='train', image_height=480, image_width=640, use_hard=True):
        self.root_dir = root_dir
        self.split = split
        self.image_height = image_height
        self.image_width = image_width
        self.use_hard = use_hard

        if os.path.exists(os.path.join(root_dir, 'hpatches', 'hpatches-release')):
            self.dataset_dir = os.path.join(root_dir, 'hpatches', 'hpatches-release')
        elif os.path.exists(os.path.join(root_dir, 'hpatches-release')):
            self.dataset_dir = os.path.join(root_dir, 'hpatches-release')
        else:
            self.dataset_dir = root_dir

        self.sequences = []
        self._collect_sequences()

        np.random.seed(42)
        n_total = len(self.sequences)
        n_train = int(n_total * 0.8)
        if split == 'train':
            self.sequences = self.sequences[:n_train]
        else:
            self.sequences = self.sequences[n_train:]

        print(f"HPatches SSL {'训练' if split == 'train' else '验证'}集: {len(self.sequences)} 个序列")

    def _collect_sequences(self):
        if not os.path.exists(self.dataset_dir):
            return

        for seq_name in sorted(os.listdir(self.dataset_dir)):
            seq_path = os.path.join(self.dataset_dir, seq_name)
            if os.path.isdir(seq_path):
                ref_path = os.path.join(seq_path, 'ref.png')
                if os.path.exists(ref_path):
                    self.sequences.append({
                        'name': seq_name,
                        'path': seq_path,
                        'type': 'illumination' if seq_name.startswith('i_') else 'viewpoint'
                    })

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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_path = seq['path']

        ref_img = cv2.imread(os.path.join(seq_path, 'ref.png'))
        if ref_img is None:
            ref_img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        ref_img = cv2.resize(ref_img, (self.image_width, self.image_height))

        img_index = np.random.randint(1, 6)
        transform_type = 'h' if self.use_hard else 'e'
        warp_img_path = os.path.join(seq_path, f'{transform_type}{img_index}.png')

        if os.path.exists(warp_img_path):
            warp_img = cv2.imread(warp_img_path)
            warp_img = cv2.resize(warp_img, (self.image_width, self.image_height))
        else:
            warp_img = ref_img.copy()

        keypoints = self._extract_keypoints(ref_img)
        labels = self._create_labels(keypoints, self.image_height, self.image_width)

        ref_tensor = torch.from_numpy(ref_img).float().permute(2, 0, 1) / 255.0
        warp_tensor = torch.from_numpy(warp_img).float().permute(2, 0, 1) / 255.0

        ref_gray = ref_tensor.mean(dim=0, keepdim=True)
        warp_gray = warp_tensor.mean(dim=0, keepdim=True)
        label_tensor = torch.from_numpy(labels).long()

        return ref_gray, warp_gray, label_tensor


def get_hpatches_dataloader(root_dir, batch_size=4, num_workers=4,
                           split='train', use_hard=True, image_height=480, image_width=640):
    from torch.utils.data import DataLoader

    dataset = HPatchesDatasetSSL(
        root_dir=root_dir,
        split=split,
        use_hard=use_hard,
        image_height=image_height,
        image_width=image_width
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )

    return loader
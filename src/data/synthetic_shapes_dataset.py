"""
Synthetic Shapes 数据集

用于接近 SuperPoint / MagicPoint 原始路线的第一阶段训练：
先在干净的几何图形上训练 detector，让网络学会什么是稳定、明确的角点。

当前版本生成的图形包括：
- 矩形
- 三角形
- 多边形
- 折线/线段交叉

输出：
- image: [1, H, W] 灰度图
- paired_image: [1, H, W]，当前阶段与 image 相同，仅为了兼容现有训练入口
- labels: [H/8, W/8]，标准 65 类 SuperPoint 标签
"""

import math
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticShapesDataset(Dataset):
    def __init__(self, num_samples=20000, image_height=480, image_width=640, grid_size=8):
        self.num_samples = num_samples
        self.image_height = image_height
        self.image_width = image_width
        self.grid_size = grid_size

    def __len__(self):
        return self.num_samples

    def _blank_canvas(self):
        return np.zeros((self.image_height, self.image_width), dtype=np.uint8)

    def _clip_point(self, x, y):
        x = int(np.clip(x, 0, self.image_width - 1))
        y = int(np.clip(y, 0, self.image_height - 1))
        return x, y

    def _draw_rectangle(self, canvas, keypoints):
        w = random.randint(40, 180)
        h = random.randint(40, 180)
        x1 = random.randint(20, max(21, self.image_width - w - 20))
        y1 = random.randint(20, max(21, self.image_height - h - 20))
        x2, y2 = x1 + w, y1 + h
        thickness = random.randint(1, 3)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, thickness)
        keypoints.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    def _draw_triangle(self, canvas, keypoints):
        pts = []
        for _ in range(3):
            x = random.randint(20, self.image_width - 21)
            y = random.randint(20, self.image_height - 21)
            pts.append([x, y])
        pts_np = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_np], True, 255, random.randint(1, 3))
        keypoints.extend([(int(x), int(y)) for x, y in pts])

    def _draw_polygon(self, canvas, keypoints):
        cx = random.randint(80, self.image_width - 81)
        cy = random.randint(80, self.image_height - 81)
        n = random.randint(4, 6)
        r = random.randint(30, 90)
        pts = []
        base = random.uniform(0, math.pi)
        for i in range(n):
            ang = base + 2 * math.pi * i / n + random.uniform(-0.15, 0.15)
            rr = r + random.randint(-10, 10)
            x = cx + rr * math.cos(ang)
            y = cy + rr * math.sin(ang)
            pts.append(self._clip_point(x, y))
        pts_np = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_np], True, 255, random.randint(1, 3))
        keypoints.extend(pts)

    def _draw_polyline(self, canvas, keypoints):
        n = random.randint(3, 6)
        pts = []
        x = random.randint(20, self.image_width - 21)
        y = random.randint(20, self.image_height - 21)
        pts.append((x, y))
        for _ in range(n - 1):
            x += random.randint(-120, 120)
            y += random.randint(-120, 120)
            pts.append(self._clip_point(x, y))
        pts_np = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_np], False, 255, random.randint(1, 3))
        keypoints.extend(pts)

    def _draw_cross(self, canvas, keypoints):
        cx = random.randint(40, self.image_width - 41)
        cy = random.randint(40, self.image_height - 41)
        dx = random.randint(20, 60)
        dy = random.randint(20, 60)
        thickness = random.randint(1, 3)
        cv2.line(canvas, (cx - dx, cy), (cx + dx, cy), 255, thickness)
        cv2.line(canvas, (cx, cy - dy), (cx, cy + dy), 255, thickness)
        keypoints.append((cx, cy))

    def _generate_shapes_image(self):
        canvas = self._blank_canvas()
        keypoints = []

        shape_fns = [
            self._draw_rectangle,
            self._draw_triangle,
            self._draw_polygon,
            self._draw_polyline,
            self._draw_cross,
        ]

        num_shapes = random.randint(6, 14)
        for _ in range(num_shapes):
            random.choice(shape_fns)(canvas, keypoints)

        if random.random() < 0.5:
            for _ in range(random.randint(3, 8)):
                x1 = random.randint(0, self.image_width - 1)
                y1 = random.randint(0, self.image_height - 1)
                x2 = random.randint(0, self.image_width - 1)
                y2 = random.randint(0, self.image_height - 1)
                cv2.line(canvas, (x1, y1), (x2, y2), random.randint(80, 180), 1)

        return canvas, keypoints

    def _augment(self, image):
        img = image.astype(np.float32)

        if random.random() < 0.8:
            img *= random.uniform(0.7, 1.3)

        if random.random() < 0.7:
            noise = np.random.normal(0, random.uniform(3, 12), img.shape)
            img += noise

        if random.random() < 0.4:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _deduplicate_keypoints(self, keypoints):
        unique = []
        occupied = set()
        for x, y in keypoints:
            x, y = self._clip_point(x, y)
            cell = (x // self.grid_size, y // self.grid_size)
            if cell in occupied:
                continue
            occupied.add(cell)
            unique.append((x, y))
        return unique

    def _create_labels(self, keypoints):
        labels = np.full(
            (self.image_height // self.grid_size, self.image_width // self.grid_size),
            64,
            dtype=np.int64,
        )

        for x, y in keypoints:
            gx = x // self.grid_size
            gy = y // self.grid_size
            if 0 <= gx < labels.shape[1] and 0 <= gy < labels.shape[0]:
                cell_x = x % self.grid_size
                cell_y = y % self.grid_size
                cls = cell_y * self.grid_size + cell_x
                labels[gy, gx] = cls

        return labels

    def __getitem__(self, idx):
        image, keypoints = self._generate_shapes_image()
        keypoints = self._deduplicate_keypoints(keypoints)
        image = self._augment(image)
        labels = self._create_labels(keypoints)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        label_tensor = torch.from_numpy(labels).long()

        return image_tensor, image_tensor.clone(), label_tensor


def get_synthetic_shapes_dataloader(batch_size=8, num_workers=0, split='train', image_height=480, image_width=640, grid_size=8):
    if split == 'train':
        num_samples = 20000
    else:
        num_samples = 2000

    dataset = SyntheticShapesDataset(
        num_samples=num_samples,
        image_height=image_height,
        image_width=image_width,
        grid_size=grid_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
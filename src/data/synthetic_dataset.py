"""
合成的SuperPoint训练数据生成器

用于训练的数据是合成生成的，包含:
- 网格图案 (规则排列的点)
- 椭圆形图案
- 线条图案

这些合成数据用于验证训练流程，在真实场景中应该使用真实图像数据。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class SyntheticDataset(Dataset):
    """
    合成数据集生成器

    生成包含规则排列关键点的合成图像，模拟真实世界的关键点分布。
    关键点以网格形式排列，便于验证模型是否学到了重复性检测能力。
    """

    def __init__(self, num_samples=1000, image_height=480, image_width=640, grid_size=8):
        """
        Args:
            num_samples: 生成的样本数量
            image_height: 图像高度
            image_width: 图像宽度
            grid_size: 关键点网格间距
        """
        self.num_samples = num_samples
        self.image_height = image_height
        self.image_width = image_width
        self.grid_size = grid_size

    def __len__(self):
        return self.num_samples

    def _generate_grid_points(self, add_noise=True):
        """
        生成网格形式排列的关键点

        关键点分布在图像的grid_size步长位置上，
        添加随机扰动模拟真实场景中的位置变化。
        """
        points = []
        h_range = range(self.grid_size, self.image_height - self.grid_size, self.grid_size)
        w_range = range(self.grid_size, self.image_width - self.grid_size, self.grid_size)

        for y in h_range:
            for x in w_range:
                if add_noise:
                    # 添加小随机扰动 (±2像素)
                    nx = np.random.randint(-2, 3)
                    ny = np.random.randint(-2, 3)
                    x_new = max(0, min(self.image_width - 1, x + nx))
                    y_new = max(0, min(self.image_height - 1, y + ny))
                else:
                    x_new, y_new = x, y
                points.append((x_new, y_new))

        return np.array(points)

    def _add_noise(self, image):
        """
        向图像添加噪声和变换，增加数据多样性
        """
        # 添加高斯噪声
        noise = np.random.normal(0, 10, image.shape).astype(np.float32)
        image = image.astype(np.float32) + noise
        image = np.clip(image, 0, 255).astype(np.uint8)

        # 随机亮度调整
        brightness = np.random.uniform(0.7, 1.3)
        image = (image * brightness).astype(np.uint8)

        return image

    def _draw_keypoints_on_image(self, image, keypoints, radius=3):
        """
        将关键点绘制到图像上

        Args:
            image: 背景图像
            keypoints: 关键点坐标列表 [(x, y), ...]
            radius: 绘制圆的半径
        """
        for (x, y) in keypoints:
            cv2.circle(image, (int(x), int(y)), radius, 255, -1)
        return image

    def __getitem__(self, idx):
        """
        生成一个训练样本

        Returns:
            image: 灰度图像 tensor [1, H, W]
            keypoints: 关键点坐标 [N, 2] (x, y格式)
            heatmap: 关键点热力图 [H/8, W/8]
        """
        # 创建黑色背景
        image = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        # 生成网格关键点
        keypoints = self._generate_grid_points(add_noise=True)

        # 在图像上绘制关键点（小圆点）
        image = self._draw_keypoints_on_image(image, keypoints)

        # 添加噪声和变换
        image = self._add_noise(image)

        # 转换为PyTorch张量 [1, H, W]
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0

        # 生成下采样8倍后的热力图GT
        heatmap = np.zeros((self.image_height // 8, self.image_width // 8), dtype=np.float32)
        for (x, y) in keypoints:
            gx, gy = x // 8, y // 8
            if 0 <= gx < heatmap.shape[1] and 0 <= gy < heatmap.shape[0]:
                heatmap[int(gy), int(gx)] = 1.0

        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # [1, H/8, W/8]

        return image_tensor, heatmap_tensor, keypoints


def visualize_sample(image, keypoints, pred_scores=None, title="Sample"):
    """
    可视化训练样本（用于调试）

    Args:
        image: 图像 tensor 或 numpy数组
        keypoints: 关键点坐标
        pred_scores: 预测分数（可选）
        title: 窗口标题
    """
    import matplotlib.pyplot as plt

    if isinstance(image, torch.Tensor):
        image = image.squeeze().numpy()

    fig, axes = plt.subplots(1, 3 if pred_scores is not None else 2, figsize=(12, 4))

    # 原图
    axes[0].imshow(image, cmap='gray')
    axes[0].scatter(keypoints[:, 0], keypoints[:, 1], s=10, c='red')
    axes[0].set_title('Input Image with Keypoints')
    axes[0].axis('off')

    # 热力图
    if isinstance(image, np.ndarray) and len(image.shape) == 2:
        axes[1].imshow(image, cmap='gray')
    axes[1].set_title('Keypoint Heatmap (GT)')
    axes[1].axis('off')

    if pred_scores is not None:
        axes[2].imshow(pred_scores.squeeze(), cmap='hot')
        axes[2].set_title('Predicted Scores')
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.close()
    print(f"Saved visualization to {title}.png")


if __name__ == "__main__":
    # 测试数据集
    dataset = SyntheticDataset(num_samples=5, image_height=480, image_width=640)

    for i in range(len(dataset)):
        image, heatmap, keypoints = dataset[i]
        print(f"Sample {i}: image shape={image.shape}, heatmap shape={heatmap.shape}, "
              f"keypoints count={len(keypoints)}")

        if i == 0:
            visualize_sample(image, keypoints, title=f'sample_{i}')
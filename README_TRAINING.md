# SuperPoint 神经网络训练指南

本指南将帮助你从零开始训练自己的 SuperPoint 关键点检测网络。

---

## 目录

1. [什么是SuperPoint？](#什么是superpoint)
2. [项目结构](#项目结构)
3. [环境配置](#环境配置)
4. [快速开始](#快速开始)
5. [训练参数说明](#训练参数说明)
6. [输出文件说明](#输出文件说明)
7. [模型架构详解](#模型架构详解)
8. [数据格式说明](#数据格式说明)
9. [常见问题](#常见问题)

---

## 什么是SuperPoint？

SuperPoint 是一种**自监督学习**的关键点检测和描述子提取网络，由 Magic Leap 在 2018 年提出。

### 核心能力

| 能力 | 说明 |
|------|------|
| **关键点检测** | 自动找出图像中最有特色的位置（角点、斑点等） |
| **描述子提取** | 为每个关键点生成一个 128 维的特征向量 |
| **跨图像匹配** | 用描述子在不同图像中匹配同一位置 |

### 应用场景

- 视觉定位 (Visual Localization)
- 增强现实 (AR) 物体追踪
- 图像拼接 (Image Stitching)
- 机器人导航
- 3D 重建

---

## 项目结构

```
neural_network_training/
├── config/
│   └── config.py          # 配置文件（超参数）
├── src/
│   ├── models/
│   │   ├── superpoint.py   # SuperPoint 网络架构
│   │   └── losses.py       # 损失函数定义
│   └── data/
│       └── synthetic_dataset.py  # 合成数据生成器
├── checkpoints/            # 训练保存的模型文件
│   ├── superpoint_epoch_5.pth
│   └── superpoint_final.pth
├── train.py                # 训练脚本（主入口）
├── train_multi_gpu.py      # 多GPU训练脚本（双卡4090优化）
├── test_install.py         # 环境测试脚本
└── README_TRAINING.md      # 本文档
```

---

## 环境配置

### 系统要求

- Python 3.9 或更高版本
- PyTorch 2.0+
- NumPy
- OpenCV (opencv-python)
- Matplotlib
- tqdm

### 安装命令

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm
```

### 验证安装

```bash
python test_install.py
```

预期输出：
```
Python: 3.10.x
PyTorch: 2.x.x
NumPy: 1.x.x
OpenCV: 4.x.x
```

---

## 快速开始

### 1. 运行训练（使用默认参数）

```bash
python train.py
```

### 2. 自定义训练参数

```bash
# 训练10个epoch，批次大小4
python train.py --epochs 10 --batch_size 4

# 使用CPU训练，设置学习率
python train.py --epochs 20 --batch_size 8 --lr 0.0005 --device cpu
```

### 3. 查看帮助

```bash
python train.py --help
```

---

## 服务器使用流程（实验室4090服务器）

### 前提条件

1. 已拥有个人账户
2. 已连接实验室 WiFi
3. VSCode 已安装 SSH 插件

### 连接步骤

#### 1. SSH 连接服务器

在本地终端或 VSCode 终端中运行：

```bash
ssh your_username@server_ip
```

例如：
```bash
ssh zhangsan@192.168.1.100
```

#### 2. 使用 VSCode SSH 插件（推荐）

1. 安装 VSCode 插件：**Remote - SSH**
2. 按 `F1`，输入 `Remote-SSH: Connect to Host`
3. 输入服务器地址 `your_username@server_ip`
4. 输入密码即可连接

#### 3. 在服务器上运行训练

连接成功后，在 VSCode 终端中：

```bash
# 进入项目目录
cd ~/neural_network_training

# 激活conda环境（如有）
conda activate your_env

# 开始双卡训练
python train_multi_gpu.py --epochs 200

# 或单卡训练
python train.py --epochs 100 --device cuda
```

### 常用命令

| 命令 | 说明 |
|------|------|
| `nvidia-smi` | 查看 GPU 状态 |
| `nvidia-smi -l 1` | 实时监控 GPU（每秒刷新） |
| `watch -n 1 nvidia-smi` | 持续监控 GPU |
| `conda info --envs` | 查看 conda 环境 |
| `squeue` | 查看 SLURM 任务队列（如有） |
| `scp file user@ip:/path/` | 传输文件到服务器 |

### GPU 监控

训练过程中实时查看 GPU 使用情况：

```bash
# 终端实时监控
watch -n 1 nvidia-smi

# 或使用 Python 版监控（更详细）
pip install gpustat
gpustat -cpu -i 1
```

### 注意事项

1. **训练时确保网络不断开** - 长时间训练建议用 `screen` 或 `tmux`
2. **不要在公共目录操作** - 使用个人目录 `~/`
3. **模型文件较大** - 定期清理 `checkpoints/` 目录

### 后台训练（网络断开不中断）

使用 `screen` 保证训练在后台持续运行：

```bash
# 创建名为 train 的会话
screen -S train

# 启动训练
cd ~/neural_network_training
python train_multi_gpu.py --epochs 200

# 按 `Ctrl+A`，然后按 `D` 暂时离开会话
# 训练继续在后台运行

# 重新连接会话
screen -r train
```

---

## 多GPU训练（双卡4090优化）

如果你有**两张4090显卡**的服务器，可以使用专用的多GPU训练脚本来大幅加速训练。

### 训练脚本

| 脚本 | 说明 |
|------|------|
| `train.py` | 普通训练脚本（单卡或自动检测） |
| `train_multi_gpu.py` | **双卡4090专用训练脚本**（推荐） |

### 双卡4090训练

```bash
python train_multi_gpu.py --epochs 200
```

### 配置说明

在 `train_multi_gpu.py` 文件顶部可以调整以下参数：

```python
# ============================================================
# 单卡训练配置 - 设为 False 则只使用单张GPU
# ============================================================
USE_MULTI_GPU = True   # True: 使用双卡, False: 只使用单卡

# ============================================================
# 训练超参数 - 可根据显存调整
# ============================================================
BATCH_SIZE_PER_GPU = 16      # 每张卡的批次大小 (4090 24GB建议16-24)
LEARNING_RATE = 0.001        # 学习率
NUM_EPOCHS = 200             # 训练轮数
USE_AMP = True               # 使用混合精度训练，省显存且加速
```

### 只使用单卡训练

如需在双卡机器上只用单卡训练，修改 `train_multi_gpu.py` 第 8 行：

```python
USE_MULTI_GPU = False  # 改为 False 即可只用单卡
```

### 多GPU训练优势

| 配置 | 单卡4090 | 双卡4090 (DataParallel) |
|------|----------|--------------------------|
| 批次大小 | 16 | 32 |
| 训练速度 | 1x | ~1.8x |
| 显存占用 | 约18GB | 每卡约15GB |

### 混合精度训练 (AMP)

`train_multi_gpu.py` 默认开启混合精度训练：
- **优点**: 减少显存占用、加速训练
- **原理**: 使用 FP16 计算，FP32 累积梯度
- **效果**: 批次大小可增加 50-100%

### 运行前检查

确保服务器已正确配置 CUDA：

```bash
nvidia-smi
```

预期输出应显示两张 4090：

```
+------------------------------------------------------------------+
| NVIDIA-SMI 525.xx       Driver Version: 525.xx     CUDA Version: 12.0     |
|--------------------+--------+----------------------+--------+
| GPU  Name          | Temp   |  Fan   | Perf   |  Memory |
|--------------------+--------+----------------------+--------+
|   0  NVIDIA GeForce.|  45°C  |   0%   |  N/A   |  24576MiB |
|   1  NVIDIA GeForce.|  42°C  |   0%   |  N/A   |  24576MiB |
+------------------------------------------------------------------+
```

---

## 训练参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--epochs` | 训练轮数 | 100 | `--epochs 50` |
| `--batch_size` | 每批次样本数 | 4 | `--batch_size 8` |
| `--lr` | 学习率 | 0.001 | `--lr 0.0005` |
| `--checkpoint_dir` | 模型保存目录 | checkpoints | `--checkpoint_dir my_models` |
| `--device` | 计算设备 | cuda 或 cpu | `--device cpu` |

### 推荐配置

| 硬件 | 批次大小 | 学习率 | Epochs |
|------|----------|--------|--------|
| CPU | 2-4 | 0.001 | 50-100 |
| GPU (4GB) | 8-16 | 0.001 | 100-200 |
| GPU (8GB+) | 16-32 | 0.001 | 100-300 |

---

## 输出文件说明

### 模型检查点 (Checkpoints)

训练过程中会保存在 `checkpoints/` 目录下：

| 文件名 | 说明 |
|--------|------|
| `superpoint_epoch_N.pth` | 第 N 个 epoch 的中间模型 |
| `superpoint_final.pth` | 训练结束后的最终模型 |

### 检查点内容

每个 `.pth` 文件包含：
- `epoch`: 训练轮数
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `loss`: 最终损失值

### 加载模型

```python
import torch
from src.models.superpoint import SuperPoint

# 加载
checkpoint = torch.load('checkpoints/superpoint_final.pth')
model = SuperPoint()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 模型架构详解

### 整体结构

```
输入图像 (1 x H x W)
    │
    ▼
┌─────────────────────────────┐
│     共享编码器 (Encoder)     │  VGG风格CNN，下采样8倍
│  1→64→64→128→256 channels    │
└─────────────────────────────┘
    │
    ├──► 检测头 (Detection Head)
    │         256→256→65 channels
    │         输出: 1 x H/8 x W/8 (关键点概率)
    │
    └──► 描述子头 (Descriptor Head)
              256→256→128 channels
              输出: 128 x H/8 x W/8 (描述子，L2归一化)
```

### 各模块功能

#### 1. 共享编码器

| 层 | 输入通道 | 输出通道 | 输出尺寸 |
|----|----------|----------|----------|
| Conv1 + Pool | 1 | 64 | H/2 x W/2 |
| Conv2 + Pool | 64 | 64 | H/4 x W/4 |
| Conv3 + Pool | 64 | 128 | H/8 x W/8 |
| Conv4 | 128 | 256 | H/8 x W/8 |

#### 2. 检测头

- 输入: 256 x H/8 x W/8
- 输出: 1 x H/8 x W/8（每个位置是关键点的概率）

#### 3. 描述子头

- 输入: 256 x H/8 x W/8
- 输出: 128 x H/8 x W/8（L2归一化的描述子）

### 模型参数量

| 模块 | 参数量 |
|------|--------|
| 编码器 | ~2.1M |
| 检测头 | ~200K |
| 描述子头 | ~200K |
| **总计** | **~2.4M** |

---

## 数据格式说明

### 当前实现：合成数据

本项目使用合成数据进行训练演示。合成数据生成规则：

1. **图像尺寸**: 480 x 640（灰度图）
2. **关键点分布**: 网格形式，间距 8 像素
3. **关键点表示**: 白色小圆点 (半径3像素)

### 合成数据格式

```python
# DataLoader 返回的每个样本:
image, heatmap, keypoints = dataset[i]

# image: torch.Tensor [1, 480, 640]
#   - 归一化的灰度图像，值范围 [0, 1]

# heatmap: torch.Tensor [1, 60, 80]
#   - 关键点热力图，下采样8倍
#   - 关键点位置值为 1.0，其他位置为 0.0

# keypoints: numpy.ndarray [N, 2]
#   - 关键点坐标 (x, y) 格式
#   - N 是该图像中的关键点数量
```

### 真实数据替换

如需使用真实数据，修改 `src/data/synthetic_dataset.py` 中的 `SyntheticDataset` 类：

```python
class SyntheticDataset(Dataset):
    def __getitem__(self, idx):
        # 替换为加载真实图像的代码
        image = load_your_image(idx)      # [H, W] 灰度图
        keypoints = load_your_keypoints(idx)  # [[x1,y1], [x2,y2], ...]

        # 转换为tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0

        # 生成热力图（下采样8倍）
        heatmap = create_heatmap(keypoints, h=60, w=80)

        return image_tensor, heatmap_tensor, keypoints
```

推荐的真实数据集：
- HPatches
- COCO
- 7-Scenes

---

## 损失函数说明

训练使用两个损失函数的组合：

### 1. 检测损失 (Detection Loss)

- **类型**: 加权交叉熵损失
- **作用**: 让模型预测的关键点位置接近真实位置
- **权重**: 正样本（关键点）权重 10.0，负样本权重 1.0

### 2. 描述子损失 (Descriptor Loss)

- **类型**: 对比损失
- **作用**: 使匹配的关键点描述子更相似，不匹配的更不同
- **边缘**: 0.5

### 总损失

```
Total_Loss = 1.0 × Detection_Loss + 0.5 × Descriptor_Loss
```

---

## 训练输出解读

### 典型训练输出

```
Epoch 1/100 | Loss: 0.5432 (Det: 0.4321, Desc: 0.1111) | LR: 0.001000 | Time: 12.5s
```

| 字段 | 说明 |
|------|------|
| `Loss` | 总损失 = 检测损失 + 0.5×描述子损失 |
| `Det` | 检测损失（关键点定位的准确性） |
| `Desc` | 描述子损失（描述子的质量） |
| `LR` | 当前学习率 |
| `Time` | 该epoch耗时 |

### 损失下降规律

| 阶段 | 检测损失 | 描述子损失 | 说明 |
|------|----------|------------|------|
| 初期 | 高 (~0.5+) | 高 | 模型在随机初始化 |
| 中期 | 下降 (0.1-0.3) | 下降 | 模型开始学习 |
| 后期 | 稳定 (~0.05-0.1) | 稳定 | 模型收敛 |

---

## 常见问题

### Q1: 训练太慢怎么办？

- 减少 `--batch_size`（但会影响收敛稳定性）
- 使用 GPU 加速训练
- 减少 `--epochs`（但模型效果会变差）

### Q2: 损失不下降怎么办？

1. 检查学习率是否太高/太低
2. 检查数据是否正确加载
3. 增加训练 epochs

### Q3: 如何使用自己的数据集？

修改 `src/data/synthetic_dataset.py`，参考本文档「真实数据替换」部分。

### Q4: 如何加载训练好的模型？

```python
import torch
from src.models.superpoint import SuperPoint

# 加载检查点
checkpoint = torch.load('checkpoints/superpoint_final.pth', map_location='cpu')

# 创建模型并加载权重
model = SuperPoint()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 使用模型
with torch.no_grad():
    descriptors, scores = model(image_tensor)
```

### Q5: 如何调整模型超参数？

编辑 `config/config.py` 文件：

```python
class Config:
    IMAGE_HEIGHT = 480      # 图像高度
    IMAGE_WIDTH = 640        # 图像宽度
    ENCODER_DIM = 256        # 编码器维度
    GRID_SIZE = 8            # 下采样倍率
    BATCH_SIZE = 4           # 批次大小
    NUM_EPOCHS = 100         # 训练轮数
    LEARNING_RATE = 0.001   # 学习率
```

---

## 后续步骤

训练完成后，你可以：

1. **可视化结果** - 使用 `src/data/synthetic_dataset.py` 中的 `visualize_sample()` 函数
2. **测试模型** - 编写推理脚本测试关键点检测效果
3. **模型转换** - 导出为 ONNX 格式用于其他平台
4. **应用集成** - 将模型集成到你的应用中

---

## 参考资料

- Original Paper: [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)
- Magic Leap GitHub: https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork

---

如有问题，请检查训练输出中的错误信息，或查看本文档的「常见问题」章节。
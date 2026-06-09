# C1 训练流程文档 (从 MagicLeap 公开预训练重训 dark teacher)

## 背景

之前的 dark/ 训练尝试 (epoch_35 teacher + 各种 distillation) 都退步,根因是:
- teacher 在极暗处召回塌缩 (190/22 kp),student 蒸馏只会固化这个塌缩
- 之前 train.py 的 4-tuple 路径 (image, image_consist, soft_heatmap, label) 在我接手时被错改成 5-tuple,
  导致 soft_heatmap 实际没载入, soft_heatmap_weight 0.6 乘 0 = 0 loss, 整个 soft 蒸馏线没起作用

C1 流程不依赖 epoch_35 teacher,改从 MagicLeap 公开预训练 (`superpoint_v6_from_tf.pth`) 重新训一个
专门针对 dark/ 的 teacher,然后再走 distillation 出学生。

## 模型兼容性

- MagicLeap 公开权重: `rpautrat/SuperPoint` 仓库 `weights/superpoint_v6_from_tf.pth` (MIT 协议, 5.2 MB)
- 我们的模型 (`src/models/superpoint.py`) 已重写以匹配 MagicLeap 架构:
  - 每个 conv 后接 BN
  - encoder 输出 128 通道 (不是 256)
  - det_conv1 / desc_conv1 输入 128 (直接接 encoder)
  - 提供 `load_magicpoint_weights()` 函数做命名映射
- 兼容性代价: **epoch_35 的 .pth 跟新模型不兼容** (encoder 结构不同),无法再用

## 完整流程

### Step 1: 下载 MagicLeap 公开预训练权重 (5.2 MB)

```bash
# 在你服务器 (能访问 GitHub raw 的机器) 拉:
wget -O checkpoints/magicpoint_v6_from_tf.pth \
    https://raw.githubusercontent.com/rpautrat/SuperPoint/master/weights/superpoint_v6_from_tf.pth

# 或者直接从 GitHub 下载页: https://github.com/rpautrat/SuperPoint/blob/master/weights/superpoint_v6_from_tf.pth
```

### Step 2: 生成 dark 数据的 HA 伪标签 (如果还没有)

```bash
python tools/generate_homographic_labels.py \
    --checkpoint checkpoints/superpoint_epoch_35.pth \
    --image_dir dark \
    --export_name dark_pseudo_v3 \
    --dark
```

(这一步仍用 epoch_35 当 teacher,因为它至少能在中等暗处找到关键点。等 C1 训出新 teacher 后
这一步可以用新 teacher 再做一次,得到质量更好的伪标签。)

### Step 3: C1 主训练 (从 MagicLeap 公开权重开始)

```bash
python train.py \
    --dataset pseudo_labels \
    --init_from_magicpoint checkpoints/magicpoint_v6_from_tf.pth \
    --pseudo_image_dir dark \
    --pseudo_keypoint_dir outputs/pseudo_labels/dark_pseudo_v3/keypoints \
    --pseudo_heatmap_dir outputs/pseudo_labels/dark_pseudo_v3/heatmaps \
    --enable_night_preprocess \
    --soft_heatmap_weight 0.6 \
    --soft_heatmap_warmup_epochs 2 \
    --entropy_weight 0.01 \
    --det_weight 1.0 \
    --dustbin_weight 0.1 \
    --epochs 30 \
    --batch_size 4 \
    --lr 0.001
```

参数说明:
- `--init_from_magicpoint` : 加载 MagicLeap 权重 (与 `--init_checkpoint` 互斥)
- `--enable_night_preprocess` : CLAHE + gamma 提亮预处理,让网络能看见暗处结构
- `--soft_heatmap_weight 0.6` : soft heatmap 蒸馏 (teacher 的连续 heatmap 当 soft target)
- `--soft_heatmap_warmup_epochs 2` : soft loss 前 2 个 epoch 从 0 线性增长
- `--entropy_weight 0.01` : 熵正则,防止过自信坍缩
- `--det_weight 1.0` : 标准 65 类 CE
- `--dustbin_weight 0.1` : dustbin 通道权重 (原版默认值)
- `--epochs 30` : 训 30 epoch,跟 MagicLeap MS-COCO 训练量接近
- `--lr 0.001` : MagicLeap 公开权重的 lr 范围

### Step 4: 评估新 teacher

```bash
python tools/evaluate_dark.py \
    --ckpt checkpoints/superpoint_final.pth \
    --image_dir dark \
    --num_pairs 20 \
    --output_dir outputs/eval_c1_teacher
```

对比指标:
- 平均关键点数 (期望 > epoch_35 的 190)
- Lowe 内点率 (期望 >= 90%,不要让 desc 退化)
- 重复率 (期望 >= 28%, 比 ORB 的 23% 略高)

### Step 5: 用新 teacher 蒸馏学生 (沿用你之前的命令模板)

```bash
python train.py \
    --dataset pseudo_labels \
    --init_checkpoint checkpoints/superpoint_final.pth \
    --pseudo_image_dir dark \
    --pseudo_keypoint_dir outputs/pseudo_labels/dark_pseudo_v3/keypoints \
    --pseudo_heatmap_dir outputs/pseudo_labels/dark_pseudo_v3/heatmaps \
    --enable_night_preprocess \
    --soft_heatmap_weight 0.6 \
    --soft_heatmap_warmup_epochs 2 \
    --entropy_weight 0.01 \
    --det_weight 1.0 \
    --epochs 10 \
    --batch_size 4 \
    --lr 0.0003
```

低 lr 精调,保住 C1 teacher 的 100% Lowe,只动 detector 召回。

## 新增的 CLI 参数 (本次)

| 参数 | 作用 | 默认 |
|---|---|---|
| `--init_from_magicpoint PATH` | 加载 MagicLeap 公开预训练权重 | 空 (默认行为) |
| `--enable_night_preprocess` | pseudo_labels 数据加载时先做 CLAHE+gamma 提亮 | False (原版行为) |
| `--dustbin_weight FLOAT` | 覆盖 dustbin 通道权重 | None (用 config 默认 0.1) |
| `--entropy_warmup_epochs INT` | entropy 正则 warmup epoch 数 | None (跟 soft 同步) |
| `--pseudo_heatmap_dir PATH` | soft heatmap 软标签目录 | 空 (无 soft loss) |

## 修改的文件 (本轮)

| 文件 | 改动 |
|---|---|
| `src/models/superpoint.py` | 重写以匹配 MagicLeap 架构 (加 BN, encoder 128 通道);新增 `load_magicpoint_weights()` |
| `src/models/losses.py` | 给 entropy 加 `entropy_warmup_epochs` 门控 (P3) |
| `src/data/pseudo_label_dataset.py` | 加 `enable_night_preprocess` / CLAHE / gamma 提亮 (P1) |
| `config/config.py` | 加 `DUSTBIN_WEIGHT=0.1` 字段, C1 默认值 |
| `train.py` | 加 `--init_from_magicpoint` / `--enable_night_preprocess` / `--dustbin_weight` / `--entropy_warmup_epochs` 4 个 CLI |
| `outputs/magicpoint_v6_from_tf.pth` | 下载的 MagicLeap 公开权重, 5.2 MB |

## 没动的文件 (跟原版保持一致)

- `src/data/synthetic_shapes_dataset.py` / `hpatches_dataset.py` / `coco_dataset.py` / `pseudo_label_dataset.py` 的 4-tuple 接口
- `tools/generate_homographic_labels.py` (357 行原版)
- `tools/analyze_pseudo_label_stats.py` / `compare_real_image_checkpoints*.py` / `train_magicpoint.py` / `visualize_synthetic_predictions.py`
- `visualize.py` / `train_multi_gpu.py`
- `src/data/synthetic_dataset.py` (老的 3-类合成数据, train_multi_gpu 还在用)
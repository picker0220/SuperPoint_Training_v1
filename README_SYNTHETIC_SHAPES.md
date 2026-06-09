# Synthetic Shapes 训练说明

这是接近 SuperPoint 原文路线的第一阶段：
先在干净的几何图形上训练 detector，让网络先学会“什么是明确的关键点”。

## 已接入内容

- 新数据集模块：`src/data/synthetic_shapes_dataset.py`
- 训练入口：`train.py --dataset synthetic_shapes`

## 当前生成的图形

- 矩形
- 三角形
- 多边形
- 折线 / 线段交叉

并带有轻度增强：
- 亮度变化
- 高斯噪声
- 轻微模糊

## 输出标签

使用标准 65 类 SuperPoint 标签：
- `0~63`：8x8 cell 内子位置
- `64`：dustbin，无关键点

## 建议训练命令

```bash
python train.py --dataset synthetic_shapes --epochs 20 --batch_size 8 --lr 0.001
```

如果显存较紧张，可以改成：

```bash
python train.py --dataset synthetic_shapes --epochs 20 --batch_size 4 --lr 0.001
```

## 训练目标

这一阶段不追求真实图像效果，主要看：

1. loss 是否快速稳定下降
2. 在合成图形上可视化时，关键点是否落在角、交点、折线拐点附近
3. 热图是否干净，不满天飞

## 下一步

如果这一步效果正常，下一阶段就是：

1. 用该 detector 作为初始模型
2. 在真实图像上做 homographic adaptation
3. 生成更稳定的伪标签
4. 再训练真实场景 detector

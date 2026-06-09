"""
MagicPoint / 初始 detector 训练入口

当前版本复用主训练脚本 train.py，固定数据集为 synthetic_shapes。
这样可以把“几何图形预训练阶段”和后续真实图像 detector 训练阶段分开。
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='训练 MagicPoint / 初始 detector')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备 (cuda/cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--resume', action='store_true', help='是否继续训练')
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_py = os.path.join(project_root, 'train.py')

    cmd = [
        sys.executable,
        train_py,
        '--dataset', 'synthetic_shapes',
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--device', str(args.device),
        '--checkpoint_dir', str(args.checkpoint_dir),
    ]

    if args.resume:
        cmd.append('--resume')

    print('运行命令:')
    print(' '.join(cmd))

    result = subprocess.run(cmd, cwd=project_root)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()

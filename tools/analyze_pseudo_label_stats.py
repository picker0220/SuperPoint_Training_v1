"""
分析 pseudo label 导出结果，重点检查夜间/低光图像上的标签质量。

用途：
1. 读取 outputs/pseudo_labels/<export_name>/stats.csv
2. 判断关键点是否过稀、热图是否过弱、是否明显偏向边缘区域
3. 输出简洁诊断结论，帮助决定这批伪标签能否用于后续 finetune
"""

import argparse
import csv
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='分析 pseudo label 统计结果')
    parser.add_argument('--stats_csv', type=str, required=True, help='stats.csv 路径')
    parser.add_argument('--low_kp_threshold', type=int, default=80, help='低关键点数阈值，低于此值视为过稀')
    parser.add_argument('--high_kp_threshold', type=int, default=600, help='高关键点数阈值，高于此值可能过密')
    parser.add_argument('--low_heatmap_mean', type=float, default=0.01, help='热图均值过低阈值')
    parser.add_argument('--edge_bias_threshold', type=float, default=0.65, help='边缘偏置阈值，左右或上下占比和过高时报警')
    parser.add_argument('--dark_focus', action='store_true', help='按夜间图像更严格地诊断')
    return parser.parse_args()


def load_rows(stats_csv):
    rows = []
    with open(stats_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'image': row['image'],
                'num_keypoints': int(float(row['num_keypoints'])),
                'heatmap_min': float(row['heatmap_min']),
                'heatmap_max': float(row['heatmap_max']),
                'heatmap_mean': float(row['heatmap_mean']),
                'left_ratio': float(row['left_ratio']),
                'right_ratio': float(row['right_ratio']),
                'top_ratio': float(row['top_ratio']),
                'bottom_ratio': float(row['bottom_ratio']),
            })
    return rows


def summarize(rows):
    counts = np.array([r['num_keypoints'] for r in rows], dtype=np.float32)
    means = np.array([r['heatmap_mean'] for r in rows], dtype=np.float32)
    maxs = np.array([r['heatmap_max'] for r in rows], dtype=np.float32)
    lefts = np.array([r['left_ratio'] for r in rows], dtype=np.float32)
    rights = np.array([r['right_ratio'] for r in rows], dtype=np.float32)
    tops = np.array([r['top_ratio'] for r in rows], dtype=np.float32)
    bottoms = np.array([r['bottom_ratio'] for r in rows], dtype=np.float32)

    return {
        'count_min': int(counts.min()),
        'count_mean': float(counts.mean()),
        'count_median': float(np.median(counts)),
        'count_max': int(counts.max()),
        'heatmap_mean_mean': float(means.mean()),
        'heatmap_mean_median': float(np.median(means)),
        'heatmap_max_mean': float(maxs.mean()),
        'left_mean': float(lefts.mean()),
        'right_mean': float(rights.mean()),
        'top_mean': float(tops.mean()),
        'bottom_mean': float(bottoms.mean()),
    }


def diagnose(rows, args):
    issues = []
    suggestions = []

    low_kp_rows = [r for r in rows if r['num_keypoints'] < args.low_kp_threshold]
    high_kp_rows = [r for r in rows if r['num_keypoints'] > args.high_kp_threshold]
    low_heat_rows = [r for r in rows if r['heatmap_mean'] < args.low_heatmap_mean]
    edge_bias_rows = [
        r for r in rows
        if (r['left_ratio'] + r['right_ratio'] > args.edge_bias_threshold)
        or (r['top_ratio'] + r['bottom_ratio'] > args.edge_bias_threshold)
    ]

    total = max(len(rows), 1)
    low_kp_ratio = len(low_kp_rows) / total
    high_kp_ratio = len(high_kp_rows) / total
    low_heat_ratio = len(low_heat_rows) / total
    edge_bias_ratio = len(edge_bias_rows) / total

    if args.dark_focus:
        sparse_alarm = 0.35
        weak_alarm = 0.40
    else:
        sparse_alarm = 0.50
        weak_alarm = 0.50

    if low_kp_ratio >= sparse_alarm:
        issues.append(f'过稀图像占比偏高: {len(low_kp_rows)}/{total} ({low_kp_ratio:.1%})')
        suggestions.append('降低 threshold，或减小 nms_distance，或提高 low-light consistency 的参与度')

    if high_kp_ratio >= 0.30:
        issues.append(f'过密图像占比偏高: {len(high_kp_rows)}/{total} ({high_kp_ratio:.1%})')
        suggestions.append('提高 threshold，或增大 nms_distance，避免夜间噪声点泛滥')

    if low_heat_ratio >= weak_alarm:
        issues.append(f'热图整体偏弱图像占比偏高: {len(low_heat_rows)}/{total} ({low_heat_ratio:.1%})')
        suggestions.append('考虑开启 normalize_heatmap，或调低检测阈值，或检查 teacher 在夜图上的响应能力')

    if edge_bias_ratio >= 0.35:
        issues.append(f'边缘偏置图像占比偏高: {len(edge_bias_rows)}/{total} ({edge_bias_ratio:.1%})')
        suggestions.append('重点检查点是否过度集中在亮窗、门口灯带、图像边缘等高对比区域')

    if not issues:
        suggestions.append('统计上没有明显异常，可以进入短程 finetune，但仍建议抽查 overlays 与 heatmaps_png')

    return issues, suggestions, {
        'low_kp_rows': low_kp_rows,
        'high_kp_rows': high_kp_rows,
        'low_heat_rows': low_heat_rows,
        'edge_bias_rows': edge_bias_rows,
    }


def print_examples(name, rows, limit=5):
    if not rows:
        return
    print(f'[{name}] 示例 (最多 {limit} 个):')
    for row in rows[:limit]:
        print(
            f"  - {row['image']}: kp={row['num_keypoints']}, "
            f"heat_mean={row['heatmap_mean']:.6f}, "
            f"L/R/T/B=({row['left_ratio']:.2f}, {row['right_ratio']:.2f}, {row['top_ratio']:.2f}, {row['bottom_ratio']:.2f})"
        )


def main():
    args = parse_args()

    if not os.path.exists(args.stats_csv):
        raise FileNotFoundError(f'stats.csv 不存在: {args.stats_csv}')

    rows = load_rows(args.stats_csv)
    if not rows:
        print('stats.csv 为空，没有可分析的数据')
        return

    summary = summarize(rows)
    issues, suggestions, groups = diagnose(rows, args)

    print('=' * 60)
    print('Pseudo Label 统计分析')
    print('=' * 60)
    print(f'图像数: {len(rows)}')
    print(f"关键点数 min / mean / median / max: {summary['count_min']} / {summary['count_mean']:.2f} / {summary['count_median']:.2f} / {summary['count_max']}")
    print(f"heatmap mean 平均 / 中位数: {summary['heatmap_mean_mean']:.6f} / {summary['heatmap_mean_median']:.6f}")
    print(f"heatmap max 平均: {summary['heatmap_max_mean']:.6f}")
    print(f"边缘分布均值 L/R/T/B: {summary['left_mean']:.3f} / {summary['right_mean']:.3f} / {summary['top_mean']:.3f} / {summary['bottom_mean']:.3f}")
    print()

    if issues:
        print('发现的问题:')
        for item in issues:
            print(f'  - {item}')
    else:
        print('发现的问题:')
        print('  - 暂无显著统计异常')

    print()
    print('建议:')
    for item in suggestions:
        print(f'  - {item}')

    print()
    print_examples('过稀图像', groups['low_kp_rows'])
    print_examples('过密图像', groups['high_kp_rows'])
    print_examples('热图偏弱图像', groups['low_heat_rows'])
    print_examples('边缘偏置图像', groups['edge_bias_rows'])


if __name__ == '__main__':
    main()

"""
SuperPoint训练配置文件

C1 流程 (从 MagicLeap 公开预训练权重重训 dark teacher) 默认配置:
- LR 0.001 (训新 teacher, 比精调 0.0003 稍大)
- DESC_LOSS_WEIGHT = 0.0 (默认不训描述子, 跟原版一致)
- DUSTBIN_WEIGHT = 0.1 (跟原版一致, 但暴露到 CLI 可调)

C1 训练命令(参考):
    python train.py --dataset pseudo_labels \\
        --init_from_magicpoint outputs/magicpoint_v6_from_tf.pth \\
        --pseudo_image_dir dataset/dark \\
        --pseudo_keypoint_dir outputs/pseudo_labels/dark_pseudo_v3/keypoints \\
        --pseudo_heatmap_dir outputs/pseudo_labels/dark_pseudo_v3/heatmaps \\
        --enable_night_preprocess \\
        --soft_heatmap_weight 0.6 \\
        --entropy_weight 0.01 \\
        --det_weight 1.0 \\
        --epochs 30 --batch_size 4 --lr 0.001
"""

class Config:
    # 数据相关
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640
    KEYPOINT_THRESHOLD = 0.5

    # 模型相关
    ENCODER_DIM = 256
    GRID_SIZE = 8

    # 训练相关
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # 损失函数权重
    DET_LOSS_WEIGHT = 1.0
    DESC_LOSS_WEIGHT = 0.0
    DUSTBIN_WEIGHT = 0.1  # C1 暴露到 CLI, 通过 --dustbin_weight 调
    DESC_MARGIN = 1.0
    DESC_N_NEG = 16

    # 暗域训练相关
    USE_LOWLIGHT_AUG = False  # C1 主流程不用 (HA 后的图已经够暗)
    MAX_WARP_OFFSET = 24
    DARK_IMAGE_DIR = 'dark'
    DARK_KEYPOINT_DIR = 'outputs/pseudo_labels/dark_pseudo_v3/keypoints'
    DARK_HEATMAP_DIR = 'outputs/pseudo_labels/dark_pseudo_v3/heatmaps'

    # 其他
    DEVICE = "cuda"
    NUM_WORKERS = 4
    PRINT_INTERVAL = 10
    SAVE_INTERVAL = 2
    CHECKPOINT_DIR = "checkpoints"

    RESUME_CHECKPOINT = ''
# VoteNet config for MiniSUNRGBD dataset - Finetune from SUNRGBD pretrained model
# 8-class desktop objects detection
# 只加载 backbone 预训练权重，bbox_head 保持随机初始化

_base_ = [
    './_base_/datasets/mini_sunrgbd_3d.py', './_base_/models/votenet.py',
    './_base_/schedules/schedule-3x.py', './_base_/default_runtime.py'
]

# 从 SUNRGBD 10类预训练模型微调 - backbone 权重迁移
# SUNRGBD 预训练权重路径
pretrained = 'checkpoints/votenet/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth'

# 加载预训练模型（用于权重迁移）
load_from = pretrained

# Model settings - 8 classes for MiniSUNRGBD
model = dict(
    # Backbone 使用预训练权重
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    bbox_head=dict(
        type='VoteHead',
        num_classes=8,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=8,
            num_dir_bins=12,
            with_rot=True,
            # Corrected mean_sizes: [l, w, h]
            mean_sizes=[
                [0.221, 0.504, 0.125],    # keyboard (0)
                [0.369, 0.418, 0.238],    # laptop (1)
                [0.258, 0.281, 0.098],    # book (2)
                [0.133, 0.134, 0.158],    # cup (3)
                [0.122, 0.119, 0.137],    # mug (4)
                [0.093, 0.137, 0.064],    # pen (5)
                [0.276, 0.290, 0.181],    # notebook (6)
                [0.244, 0.308, 0.175],    # phone (7)
            ],
        ),
    ),
)

# Auto scale LR for smaller dataset
auto_scale_lr = dict(enable=True, base_batch_size=128)

# 训练输出目录
work_dir = 'work_dirs/votenet_mini_sunrgbd_finetune'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

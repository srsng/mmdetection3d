# VoteNet config for MiniSUNRGBD dataset - Finetune from SUNRGBD pretrained model
# 5-class desktop objects detection (keyboard, laptop, book, cup, mug)
# 只加载 backbone 预训练权重，bbox_head 保持随机初始化

_base_ = [
    "./_base_/datasets/mini_sunrgbd_3d.py",
    "./_base_/models/votenet.py",
    "./_base_/schedules/schedule-3x.py",
    "./_base_/default_runtime.py",
]

# 从 SUNRGBD 10类预训练模型微调 - backbone 权重迁移
# SUNRGBD 预训练权重路径
pretrained = (
    "checkpoints/votenet/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth"
)

# 加载预训练模型（用于权重迁移）
load_from = pretrained

# Model settings - 5 classes for MiniSUNRGBD
# Classes: book(0), cup(1), keyboard(2), laptop(3), mug(4)
model = dict(
    # Backbone 使用预训练权重
    backbone=dict(init_cfg=dict(type="Pretrained", checkpoint=pretrained)),
    bbox_head=dict(
        type="VoteHead",
        num_classes=5,
        bbox_coder=dict(
            type="PartialBinBasedBBoxCoder",
            num_sizes=5,
            num_dir_bins=12,
            with_rot=True,
            # Mean sizes for 5 MiniSUNRGBD classes
            # Format: [length, width, height] in meters
            mean_sizes=[
                [0.255140, 0.272017, 0.111679],  # book (0)
                [0.129553, 0.131255, 0.151869],  # cup (1)
                [0.208131, 0.504047, 0.115639],  # keyboard (2)
                [0.356808, 0.411205, 0.228216],  # laptop (3)
                [0.138085, 0.127501, 0.143335],  # mug (4)
            ],
        ),
    ),
)

# Auto scale LR for smaller dataset
auto_scale_lr = dict(enable=True, base_batch_size=128)

# 训练输出目录
work_dir = "work_dirs/votenet_mini_sunrgbd_finetune"

vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

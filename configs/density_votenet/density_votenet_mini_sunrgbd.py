# VoteNet with Density-aware PointNet2 and CGNL Neck
# MiniSUNRGBD dataset (5-class desktop objects)

_base_ = [
    '../_base_/datasets/mini_sunrgbd_3d.py',
    '../_base_/models/votenet.py',
    '../_base_/schedules/schedule-3x.py',
    '../_base_/default_runtime.py',
]

# Model settings - 5 classes for MiniSUNRGBD
model = dict(
    backbone=dict(
        type='DensityAwarePointNet2',
        in_channels=5,  # xyz + height (shift_height) + density
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256),
                       (128, 128, 256), (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
    ),
    neck=dict(
        type='CGNLLocalFusionNeck',
        in_channels=256,  # FP output channels from DensityAwarePointNet2
        num_blocks=1,
        groups=4,
        reduction=4,
    ),
    bbox_head=dict(
        num_classes=5,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=5,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[
                [0.208, 0.504, 0.116],    # keyboard (0)
                [0.357, 0.411, 0.228],    # laptop (1)
                [0.255, 0.272, 0.112],    # book (2)
                [0.130, 0.131, 0.152],    # cup (3)
                [0.138, 0.128, 0.143],    # mug (4)
            ]),
    ))

# Training dataloader
train_dataloader = dict(
    batch_size=20,
    dataset=dict(
        dataset=dict(
            data_root='data2/mini_sunrgbd/')))

# Validation dataloader
val_dataloader = dict(
    batch_size=20,
    dataset=dict(
        data_root='data2/mini_sunrgbd/'))

# Test dataloader
test_dataloader = dict(
    batch_size=20,
    dataset=dict(
        data_root='data2/mini_sunrgbd/'))

# Auto scale LR for smaller dataset
auto_scale_lr = dict(enable=True, base_batch_size=128)

# Visualization
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=3),
)

# Work directory
work_dir = 'work_dirs/density_votenet_mini_sunrgbd'

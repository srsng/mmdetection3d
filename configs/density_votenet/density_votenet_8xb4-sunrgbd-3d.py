# VoteNet with Density-aware PointNet2 and CGNL Neck
# Full SUNRGBD dataset (10-class indoor objects)

_base_ = [
    '../_base_/datasets/sunrgbd-3d.py',
    '../_base_/models/votenet.py',
    '../_base_/schedules/schedule-3x.py',
    '../_base_/default_runtime.py',
]

# Model settings - 10 classes for SUNRGBD
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
        num_classes=10,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=10,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[
                [2.114, 2.393, 0.855],    # bed (0)
                [0.605, 0.699, 0.712],    # table (1)
                [2.644, 1.056, 0.916],    # sofa (2)
                [0.601, 0.601, 0.942],    # chair (3)
                [0.425, 0.437, 0.761],    # toilet (4)
                [0.414, 1.165, 0.738],    # desk (5)
                [1.055, 0.595, 1.022],    # dresser (6)
                [0.496, 0.509, 0.782],    # night_stand (7)
                [0.341, 2.151, 1.836],     # bookshelf (8)
                [1.321, 0.681, 0.582],     # bathtub (9)
            ]),
    ))

# Training dataloader - use full SUNRGBD dataset
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        dataset=dict(
            data_root='data2/sunrgbd/')))

# Validation dataloader
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_root='data2/sunrgbd/'))

# Test dataloader
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_root='data2/sunrgbd/'))

# Auto scale LR for dataset size
auto_scale_lr = dict(enable=True, base_batch_size=128)

# Visualization
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Work directory
work_dir = 'work_dirs/density_votenet_8xb4-sunrgbd-3d'

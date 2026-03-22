# VoteNet config for MiniSUNRGBD dataset
# 8-class desktop objects detection

_base_ = [
    './_base_/datasets/mini_sunrgbd_3d.py', './_base_/models/votenet.py',
    './_base_/schedules/schedule-3x.py', './_base_/default_runtime.py'
]

# Model settings - 8 classes for MiniSUNRGBD
# Classes: keyboard(0), laptop(1), book(2), cup(3), mug(4), pen(5), notebook(6), phone(7)
model = dict(
    bbox_head=dict(
        num_classes=8,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=8,
            num_dir_bins=12,
            with_rot=True,
            # Mean sizes for 8 MiniSUNRGBD classes
            # Format: [length, width, height] in meters
            # Based on actual computed mean_sizes from the dataset
            mean_sizes=[
                [0.103, 0.210, 0.234],    # keyboard (0)
                [0.222, 0.334, 0.382],    # laptop (1)
                [0.099, 0.266, 0.236],    # book (2)
                [0.159, 0.130, 0.041],    # cup (3)
                [0.140, 0.143, -0.403],   # mug (4)
                [0.064, 0.093, 0.076],    # pen (5)
                [0.181, 0.276, -0.265],   # notebook (6)
                [0.175, 0.244, -0.203],   # phone (7)
            ]),
    ))

# Auto scale LR for smaller dataset
auto_scale_lr = dict(enable=True, base_batch_size=128)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

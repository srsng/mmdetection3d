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
            # Corrected: [l, w, h] from actual computed mean_sizes
            mean_sizes=[
                [0.221, 0.504, 0.125],    # keyboard (0)
                [0.369, 0.418, 0.238],    # laptop (1)
                [0.258, 0.281, 0.098],    # book (2)
                [0.133, 0.134, 0.158],    # cup (3)
                [0.122, 0.119, 0.137],    # mug (4)
                [0.093, 0.137, 0.064],    # pen (5)
                [0.276, 0.290, 0.181],    # notebook (6)
                [0.244, 0.308, 0.175],    # phone (7)
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

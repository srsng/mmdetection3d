# VoteNet config for MiniSUNRGBD dataset
# 5-class desktop objects detection

_base_ = [
    './_base_/datasets/mini_sunrgbd_3d.py', './_base_/models/votenet.py',
    './_base_/schedules/schedule-3x.py', './_base_/default_runtime.py'
]

# Model settings - 5 classes for MiniSUNRGBD
# Classes: keyboard(0), laptop(1), book(2), cup(3), mug(4)
model = dict(
    bbox_head=dict(
        num_classes=5,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=5,
            num_dir_bins=12,
            with_rot=True,
            # Mean sizes for 5 MiniSUNRGBD classes
            # Format: [length, width, height] in meters
            # From actual computed mean_sizes
            mean_sizes=[
                [0.208, 0.504, 0.116],    # keyboard (0)
                [0.357, 0.411, 0.228],    # laptop (1)
                [0.255, 0.272, 0.112],    # book (2)
                [0.130, 0.131, 0.152],    # cup (3)
                [0.138, 0.128, 0.143],    # mug (4)
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

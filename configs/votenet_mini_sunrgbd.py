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
                [0.128, 0.215, 0.184],    # keyboard
                [0.215, 0.349, -0.093],   # laptop
                [0.114, 0.242, 0.032],    # book
                [0.154, 0.131, -0.219],   # cup
                [0.141, 0.136, -0.325],   # mug
                [0.064, 0.093, 0.076],    # pen
                [0.181, 0.276, -0.265],   # notebook
                [0.175, 0.244, -0.203],   # phone
            ]),
    ))

# Auto scale LR for smaller dataset
auto_scale_lr = dict(enable=True, base_batch_size=128)

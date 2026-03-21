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
            # Based on typical desktop object dimensions
            mean_sizes=[
                [0.60, 0.20, 0.04],   # keyboard - 60cm x 20cm x 4cm
                [0.35, 0.25, 0.02],   # laptop - 35cm x 25cm x 2cm
                [0.25, 0.18, 0.03],   # book - 25cm x 18cm x 3cm
                [0.08, 0.08, 0.12],   # cup - 8cm diameter x 12cm height
                [0.10, 0.10, 0.12],   # mug - 10cm diameter x 12cm height
                [0.14, 0.015, 0.015], # pen - 14cm x 1.5cm x 1.5cm
                [0.30, 0.21, 0.03],   # notebook - 30cm x 21cm x 3cm
                [0.15, 0.08, 0.008],  # phone - 15cm x 8cm x 0.8cm
            ]),
    ))

# Auto scale LR for smaller dataset
auto_scale_lr = dict(enable=True, base_batch_size=128)

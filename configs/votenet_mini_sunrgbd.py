# VoteNet config for MiniSUNRGBD dataset
# 5-class desktop objects detection

_base_ = [
    "./_base_/datasets/mini_sunrgbd_3d.py",
    "./_base_/models/votenet.py",
    "./_base_/schedules/schedule-3x.py",
    "./_base_/default_runtime.py",
]

# Model settings - 5 classes for MiniSUNRGBD
# Classes: book(0), cup(1), keyboard(2), laptop(3), mug(4)
model = dict(
    bbox_head=dict(
        num_classes=5,
        bbox_coder=dict(
            type="PartialBinBasedBBoxCoder",
            num_sizes=5,
            num_dir_bins=12,
            with_rot=True,
            # Mean sizes for 5 MiniSUNRGBD classes
            # Format: [length, width, height] in meters
            # From actual computed mean_sizes
            mean_sizes=[
                [0.255140, 0.272017, 0.111679],  # book (0)
                [0.129553, 0.131255, 0.151869],  # cup (1)
                [0.208131, 0.504047, 0.115639],  # keyboard (2)
                [0.356808, 0.411205, 0.228216],  # laptop (3)
                [0.138085, 0.127501, 0.143335],  # mug (4)
            ],
        ),
    )
)

# Auto scale LR for smaller dataset
auto_scale_lr = dict(enable=True, base_batch_size=128)

vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

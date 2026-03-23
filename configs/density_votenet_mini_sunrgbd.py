# VoteNet with CGNL Neck config for MiniSUNRGBD dataset
# 5-class desktop objects detection with density fusion

_base_ = [
    './_base_/datasets/mini_sunrgbd_3d.py',
    './_base_/models/votenet.py',
    './_base_/schedules/schedule-3x.py',
    './_base_/default_runtime.py',
]

# CGNL 模块注册通过 perception.detection.models.cgnl 导入

# 数据路径修正 - 覆盖 data_root
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        dataset=dict(
            data_root='data2/mini_sunrgbd/')))
val_dataloader = dict(
    dataset=dict(
        data_root='data2/mini_sunrgbd/'))
test_dataloader = dict(
    dataset=dict(
        data_root='data2/mini_sunrgbd/'))

# 修改 model：CGNL Neck 适配 VoteNet backbone 的字典输出
model = dict(
    neck=dict(
        type='VoteNetCGNLNeckWrapper',
        in_channels=256,  # VoteNet backbone 最后 fp 层的输出通道
        out_channels=256,
        num_blocks=1,
        reduction=2,
        use_scale=True,
        target_layer_idx=-1,  # 作用于最后一层上采样特征
    ),
    bbox_head=dict(
        type='VoteHead',
        num_classes=5,  # MiniSUNRGBD 5 类 (keyboard, laptop, book, cup, mug)
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_dir_bins=12,
            num_sizes=5,
            mean_sizes=[
                [0.208, 0.504, 0.116],    # keyboard (0)
                [0.357, 0.411, 0.228],    # laptop (1)
                [0.255, 0.272, 0.112],    # book (2)
                [0.130, 0.131, 0.152],    # cup (3)
                [0.138, 0.128, 0.143],    # mug (4)
            ],
            with_rot=True,
        ),
    ),
)

# 训练输出目录
work_dir = 'work_dirs/density_votenet_mini_sunrgbd'

# 小数据集自动 scale LR
auto_scale_lr = dict(enable=True, base_batch_size=128)

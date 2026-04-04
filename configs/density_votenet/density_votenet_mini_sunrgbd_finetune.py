# VoteNet with Density-aware PointNet2 - Mini SUNRGBD Fine-tune
# 基于完整数据集预训练模型在mini数据集上进行微调

_base_ = [
    './density_votenet_mini_sunrgbd.py',
]

# 加载预训练的backbone和neck权重 (跳过bbox_head)
# 使用 extract_backbone_neck.py 从完整数据集checkpoint提取
load_from = 'work_dirs/density_votenet_8xb4-sunrgbd-3d/backbone_neck_finetune.pth'

# 微调策略:
# - backbone: 从预训练权重初始化, 需要较小学习率
# - neck: 从预训练权重初始化, 需要较小学习率
# - bbox_head: 从头训练(随机初始化)

# 学习率调整 (backbone/neck用较小LR, bbox_head用标准LR)
# 通过paramwise_cfg配置不同层的学习率
param_scheduler = [
    # Warmup阶段 (begin=0, 第一个epoch是warmup)
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, end=1),
    # 主训练阶段 (20 epoch, begin=0, end=20表示0-19epoch)
    dict(type='MultiStepLR', begin=0, end=20, milestones=[15, 18], gamma=0.1),
]

# 训练配置
train_dataloader = dict(
    batch_size=20, dataset=dict(dataset=dict(data_root='data2/mini_sunrgbd/')))

# 验证配置
val_dataloader = dict(
    batch_size=20, dataset=dict(data_root='data2/mini_sunrgbd/'))

# Test dataloader
test_dataloader = dict(
    batch_size=20, dataset=dict(data_root='data2/mini_sunrgbd/'))

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=3), )

# Work directory
work_dir = 'work_dirs/density_votenet_mini_sunrgbd_finetune'

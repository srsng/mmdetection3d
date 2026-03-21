# 密度融合 VoteNet 配置 for MiniSUNRGBD dataset
# 实现论文 "Point cloud 3D object detection method based on density information-local feature fusion"

# 自定义模块导入
custom_imports = dict(
    imports=[
        'perception.detection.models.cgnl',
        'perception.detection.models.density_votenet',
        'perception.detection.models.transforms',
        'mmdet3d.datasets.mini_sunrgbd_dataset',
        'mmdet3d.datasets.transforms',
        'mmdet3d.visualization',
    ],
    allow_failed_imports=False
)

_base_ = [
    '../_base_/datasets/mini_sunrgbd_3d_density.py',
    '../_base_/schedules/schedule-3x.py',
    '../_base_/default_runtime.py',
]


# 数据路径修正
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root='data2/mini_sunrgbd/',
        ann_file='mini_sunrgbd_infos_train.pkl',
        data_prefix=dict(pts='points'),
        box_type_3d='Depth',
        modality=dict(use_lidar=True, use_camera=False)))
val_dataloader = dict(
    dataset=dict(
        data_root='data2/mini_sunrgbd/',
        ann_file='mini_sunrgbd_infos_val.pkl',
        data_prefix=dict(pts='points'),
        box_type_3d='Depth',
        modality=dict(use_lidar=True, use_camera=False)))
test_dataloader = dict(
    dataset=dict(
        data_root='data2/mini_sunrgbd/',
        ann_file='mini_sunrgbd_infos_val.pkl',
        data_prefix=dict(pts='points'),
        box_type_3d='Depth',
        modality=dict(use_lidar=True, use_camera=False)))

# 密度融合 VoteNet 模型
# 架构：DensityPointNet2SASSG (4层SA密度融合 + 2层FP) → CGNL → VoteHead
model = dict(
    type='VoteNet',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=4,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=(
            (64, 64, 128),
            (128, 128, 256),
            (128, 128, 256),
            (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
    ),
    bbox_head=dict(
        type='VoteHead',
        num_classes=8,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=8,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[
                [0.60, 0.20, 0.04],   # keyboard
                [0.35, 0.25, 0.02],   # laptop
                [0.25, 0.18, 0.03],   # book
                [0.08, 0.08, 0.12],   # cup
                [0.10, 0.10, 0.12],   # mug
                [0.14, 0.015, 0.015], # pen
                [0.30, 0.21, 0.03],   # notebook
                [0.15, 0.08, 0.008],  # phone
            ]),
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True),
        pred_layer_cfg=dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True),
        objectness_loss=dict(
            type='mmdet.CrossEntropyLoss',
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=5.0),
        center_loss=dict(
            type='ChamferDistance',
            mode='l2',
            reduction='sum',
            loss_src_weight=10.0,
            loss_dst_weight=10.0),
        dir_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=10.0 / 3.0),
        semantic_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
    ),
    train_cfg=dict(
        pos_distance_thr=0.3,
        neg_distance_thr=0.6,
        sample_mode='vote'
    ),
    test_cfg=dict(
        sample_mode='seed',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True
    ),
)

# 训练输出目录
work_dir = 'work_dirs/density_votenet_mini_sunrgbd_v2'

# 小数据集自动 scale LR
auto_scale_lr = dict(enable=True, base_batch_size=128)

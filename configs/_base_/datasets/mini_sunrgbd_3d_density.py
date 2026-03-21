# MiniSUNRGBD dataset config with density computation
# 添加密度计算到数据管道

dataset_type = 'MiniSUNRGBDDataset'
data_root = 'data/mini_sunrgbd/'
class_names = ('keyboard', 'laptop', 'book', 'cup', 'mug',
               'pen', 'notebook', 'phone')

metainfo = dict(classes=class_names)

backend_args = None

# 训练管道：添加密度计算
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,  # 密度融合不需要height维度
        load_dim=6,
        use_dim=[0, 1, 2],  # 只使用 xyz
        backend_args=backend_args),
    dict(type='LoadAnnotations3D'),
    # 计算密度并添加到点云 (xyz -> xyzd)
    dict(type='ComputeDensity', method='knn', k=16, normalize=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=False),
    dict(type='PointSample', num_points=20000),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,  # 密度融合不需要height维度
        load_dim=6,
        use_dim=[0, 1, 2],
        backend_args=backend_args),
    # 计算密度
    dict(type='ComputeDensity', method='knn', k=16, normalize=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
            ),
            dict(type='PointSample', num_points=20000)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='mini_sunrgbd_infos_train.pkl',
        pipeline=train_pipeline,
        filter_empty_gt=False,
        metainfo=metainfo,
        box_type_3d='Depth',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='mini_sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Depth',
        backend_args=backend_args))
test_dataloader = val_dataloader
val_evaluator = dict(type='IndoorMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# GroupFree3D for TO-Scene 3D Detection (70 classes)
_base_ = [
    "../_base_/datasets/to-scene-3d.py",
    "../_base_/models/groupfree3d.py",
    "../_base_/schedules/schedule-3x.py",
    "../_base_/default_runtime.py",
]

class_names = (
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "showercurtrain",
    "toilet",
    "sink",
    "bathtub",
    "garbagebin",
    "bag",
    "bottle",
    "bowl",
    "camera",
    "can",
    "cap",
    "clock",
    "keyboard",
    "display",
    "earphone",
    "jar",
    "knife",
    "lamp",
    "laptop",
    "microphone",
    "microwave",
    "mug",
    "printer",
    "remote control",
    "phone",
    "alarm",
    "book",
    "cake",
    "calculator",
    "candle",
    "charger",
    "chessboard",
    "coffee_machine",
    "comb",
    "cutting_board",
    "dishes",
    "doll",
    "eraser",
    "eye_glasses",
    "file_box",
    "fork",
    "fruit",
    "globe",
    "hat",
    "mirror",
    "notebook",
    "pencil",
    "plant",
    "plate",
    "radio",
    "ruler",
    "saucepan",
    "spoon",
    "tea_pot",
    "toaster",
    "vase",
    "vegetables",
)

# model settings
model = dict(
    backbone=dict(
        type="PointNet2SASSG",
        in_channels=6,  # N6 xyzrgb 格式
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256), (128, 128, 256)),
        fp_channels=((256, 256), (256, 288)),
        norm_cfg=dict(type="BN2d"),
        sa_cfg=dict(
            type="PointSAModule", pool_mod="max", use_xyz=True, normalize_xyz=True
        ),
    ),
    bbox_head=dict(
        type="GroupFree3DHead",
        num_classes=70,
        size_cls_agnostic=False,
        bbox_coder=dict(
            type="GroupFree3DBBoxCoder",
            num_sizes=70,
            num_dir_bins=1,
            with_rot=False,
            size_cls_agnostic=False,
            mean_sizes=[
                [0.752241, 0.958975, 0.956157],  # cabinet (0)
                [1.784878, 1.908401, 1.130330],  # bed (1)
                [0.624935, 0.630331, 0.704542],  # chair (2)
                [1.452529, 1.497418, 0.825735],  # sofa (3)
                [0.972537, 1.039505, 0.620914],  # table (4)
                [0.579747, 0.598664, 1.753817],  # door (5)
                [1.300518, 0.799717, 1.230230],  # window (6)
                [0.853999, 1.232486, 1.664450],  # bookshelf (7)
                [0.216481, 0.552806, 0.615963],  # picture (8)
                [1.275056, 1.874345, 0.257304],  # counter (9)
                [1.037709, 1.467463, 0.862807],  # desk (10)
                [1.441599, 0.893432, 1.621769],  # curtain (11)
                [0.638814, 0.714853, 1.372790],  # refrigerator (12)
                [0.398955, 0.390511, 1.627421],  # showercurtrain (13)
                [0.602399, 0.595818, 0.764255],  # toilet (14)
                [0.511421, 0.515758, 0.282268],  # sink (15)
                [1.198284, 1.048769, 0.515744],  # bathtub (16)
                [0.599379, 0.564992, 0.623212],  # garbagebin (17)
                [0.190009, 0.194591, 0.256981],  # bag (18)
                [0.076269, 0.076212, 0.219265],  # bottle (19)
                [0.157109, 0.156620, 0.061973],  # bowl (20)
                [0.144606, 0.154886, 0.110610],  # camera (21)
                [0.074692, 0.074736, 0.125076],  # can (22)
                [0.219659, 0.212206, 0.096524],  # cap (23)
                [0.113998, 0.138426, 0.181300],  # clock (24)
                [0.274123, 0.286843, 0.029591],  # keyboard (25)
                [0.251683, 0.294724, 0.294629],  # display (26)
                [0.177926, 0.174990, 0.076050],  # earphone (27)
                [0.134209, 0.135070, 0.230804],  # jar (28)
                [0.170447, 0.134771, 0.019452],  # knife (29)
                [0.189637, 0.186330, 0.353129],  # lamp (30)
                [0.321992, 0.326434, 0.208608],  # laptop (31)
                [0.149330, 0.141895, 0.163555],  # microphone (32)
                [0.286934, 0.318380, 0.194970],  # microwave (33)
                [0.105237, 0.108960, 0.102403],  # mug (34)
                [0.317763, 0.328486, 0.173343],  # printer (35)
                [0.130715, 0.117282, 0.023502],  # remote control (36)
                [0.128879, 0.121960, 0.018638],  # phone (37)
                [0.118001, 0.135721, 0.139107],  # alarm (38)
                [0.214589, 0.207122, 0.041921],  # book (39)
                [0.254257, 0.253436, 0.214801],  # cake (40)
                [0.169467, 0.160458, 0.028406],  # calculator (41)
                [0.100207, 0.098758, 0.282354],  # candle (42)
                [0.087978, 0.088161, 0.037929],  # charger (43)
                [0.379411, 0.385097, 0.080055],  # chessboard (44)
                [0.303841, 0.279200, 0.322477],  # coffee_machine (45)
                [0.102351, 0.116364, 0.031176],  # comb (46)
                [0.295202, 0.320434, 0.056676],  # cutting_board (47)
                [0.257740, 0.265655, 0.075733],  # dishes (48)
                [0.113322, 0.127469, 0.249153],  # doll (49)
                [0.060693, 0.058048, 0.019173],  # eraser (50)
                [0.155614, 0.160131, 0.051409],  # eye_glasses (51)
                [0.267863, 0.294990, 0.275771],  # file_box (52)
                [0.140247, 0.110152, 0.023915],  # fork (53)
                [0.107237, 0.107443, 0.096212],  # fruit (54)
                [0.204358, 0.209519, 0.262091],  # globe (55)
                [0.228450, 0.222675, 0.108107],  # hat (56)
                [0.089148, 0.144515, 0.224607],  # mirror (57)
                [0.208009, 0.202571, 0.022360],  # notebook (58)
                [0.131633, 0.102514, 0.018969],  # pencil (59)
                [0.213456, 0.214487, 0.326710],  # plant (60)
                [0.231961, 0.231483, 0.023800],  # plate (61)
                [0.174917, 0.178267, 0.168156],  # radio (62)
                [0.164826, 0.143083, 0.013827],  # ruler (63)
                [0.275966, 0.274502, 0.187025],  # saucepan (64)
                [0.136401, 0.118908, 0.027131],  # spoon (65)
                [0.201563, 0.208137, 0.165064],  # tea_pot (66)
                [0.293788, 0.283936, 0.217264],  # toaster (67)
                [0.141620, 0.140554, 0.267366],  # vase (68)
                [0.195042, 0.179074, 0.080797],  # vegetables (69)
            ],
        ),
        sampling_objectness_loss=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=8.0,
        ),
        objectness_loss=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        center_loss=dict(
            type="mmdet.SmoothL1Loss", beta=0.04, reduction="sum", loss_weight=10.0
        ),
        dir_class_loss=dict(
            type="mmdet.CrossEntropyLoss", reduction="sum", loss_weight=1.0
        ),
        dir_res_loss=dict(type="mmdet.SmoothL1Loss", reduction="sum", loss_weight=10.0),
        size_class_loss=dict(
            type="mmdet.CrossEntropyLoss", reduction="sum", loss_weight=1.0
        ),
        size_res_loss=dict(
            type="mmdet.SmoothL1Loss",
            beta=1.0 / 9.0,
            reduction="sum",
            loss_weight=10.0 / 9.0,
        ),
        semantic_loss=dict(
            type="mmdet.CrossEntropyLoss", reduction="sum", loss_weight=1.0
        ),
    ),
    train_cfg=dict(sample_mode="kps"),
    test_cfg=dict(
        sample_mode="kps",
        nms_thr=0.25,
        score_thr=0.0,
        per_class_proposal=True,
        prediction_stages="last",
    ),
)

metainfo = dict(classes=class_names)
backend_args = None

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,  # N6 xyzrgb 不需要 shift_height
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],  # N6 xyzrgb，全部 6 通道
        backend_args=backend_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True,
        backend_args=backend_args,
    ),
    dict(type="GlobalAlignment", rotation_axis=2),
    dict(type="PointSample", num_points=40000),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.087266, 0.087266],  # ±15度
        scale_ratio_range=[0.9, 1.1],
        shift_height=False,
    ),
    dict(
        type="Pack3DDetInputs",
        keys=[
            "points",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "pts_semantic_mask",
            "pts_instance_mask",
        ],
    ),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],  # N6 xyzrgb
        backend_args=backend_args,
    ),
    dict(type="GlobalAlignment", rotation_axis=2),  # 保持与训练一致
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="RandomFlip3D",
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5,
            ),
            dict(type="PointSample", num_points=40000),
        ],
    ),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=5,
        dataset=dict(
            type="TOSceneDataset",
            data_root="data2/TO-SCENE-down/TO-scannet/",
            ann_file="toscene_infos_train.pkl",
            pipeline=train_pipeline,
            filter_empty_gt=False,
            metainfo=metainfo,
            box_type_3d="Depth",
            backend_args=backend_args,
        ),
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="TOSceneDataset",
        data_root="data2/TO-SCENE-down/TO-scannet/",
        ann_file="toscene_infos_val.pkl",
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d="Depth",
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="IndoorMetric")
test_evaluator = val_evaluator

# 优化器配置
lr = 0.006
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.0005),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "bbox_head.decoder_layers": dict(lr_mult=0.1, decay_mult=1.0),
            "bbox_head.decoder_self_posembeds": dict(lr_mult=0.1, decay_mult=1.0),
            "bbox_head.decoder_cross_posembeds": dict(lr_mult=0.1, decay_mult=1.0),
            "bbox_head.decoder_query_proj": dict(lr_mult=0.1, decay_mult=1.0),
            "bbox_head.decoder_key_proj": dict(lr_mult=0.1, decay_mult=1.0),
        },
    ),
)

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[44, 56, 68],
        gamma=0.2,
    )
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=80, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=10),
)

randomness = dict(seed=4)
work_dir = "work_dirs/groupfree3d_tosence"

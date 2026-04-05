# TO-Scene dataset config for 3D detection (70 classes)
# With point cloud density computation for Density-aware PointNet2

dataset_type = "TOSceneDataset"
data_root = "data2/TO-SCENE-down/TO-scannet/"

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

metainfo = dict(classes=class_names)

backend_args = None

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="DEPTH",
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
        backend_args=backend_args,
    ),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=False,
        backend_args=backend_args,
    ),
    dict(type="GlobalAlignment", rotation_axis=2),
    # Compute point cloud density for Density-aware PointNet2
    dict(
        type="ComputePointDensity",
        kernel="gaussian",
        sigma=0.1,
        k_neighbor=64,
        epsilon=1e-10,
    ),
    dict(type="PointSample", num_points=40000),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[1.0, 1.0],
        shift_height=True,
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
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
        backend_args=backend_args,
    ),
    dict(type="GlobalAlignment", rotation_axis=2),
    # Compute point cloud density for Density-aware PointNet2
    dict(
        type="ComputePointDensity",
        kernel="gaussian",
        sigma=0.1,
        k_neighbor=64,
        epsilon=1e-10,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
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
    batch_size=32,
    num_workers=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
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
    batch_size=32,
    num_workers=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
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

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

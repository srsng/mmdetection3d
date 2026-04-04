# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class MiniSUNRGBDDataset(Det3DDataset):
    """MiniSUNRGBD Dataset.

    MiniSUNRGBD is a subset of SUNRGBD containing desktop object classes.
    This dataset re-indexes classes from 0-7 for the 8 target classes:
    keyboard, laptop, book, cup, mug, pen, notebook, phone.

    Note: Inherits from Det3DDataset to avoid label_mapping conflicts
    with SUNRGBDDataset's original 10-class METAINFO. But adds parse_ann_info
    override to properly convert gt_bboxes_3d to DepthInstance3DBoxes.
    """

    METAINFO = {
        "classes": ("book", "cup", "keyboard", "laptop", "mug"),
        "palette": [
            (230, 25, 72),    # keyboard - red
            (60, 180, 75),    # laptop - green
            (255, 225, 25),   # book - yellow
            (0, 130, 200),    # cup - blue
            (245, 130, 48),   # mug - orange
        ]
    }

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        metainfo: dict = None,
        data_prefix: dict = dict(pts="points", img="sunrgbd_trainval/image"),
        pipeline: list = [],
        default_cam_key: str = "CAM0",
        modality: dict = dict(use_camera=True, use_lidar=True),
        box_type_3d: str = "Depth",
        filter_empty_gt: bool = True,
        test_mode: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            default_cam_key=default_cam_key,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs,
        )
        assert "use_camera" in self.modality and "use_lidar" in self.modality
        assert self.modality["use_camera"] or self.modality["use_lidar"]

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process
        the `instances` field to `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        if self.modality["use_lidar"]:
            import os.path as osp

            # Check if it already has points prefix
            lidar_path = info["lidar_points"]["lidar_path"]
            pts_prefix = self.data_prefix.get("pts", "")
            if pts_prefix and not lidar_path.startswith(pts_prefix):
                info["lidar_points"]["lidar_path"] = osp.join(pts_prefix, lidar_path)

        if self.modality["use_camera"]:
            import os.path as osp

            for cam_id, img_info in info["images"].items():
                if "img_path" in img_info:
                    img_path = img_info["img_path"]
                    img_prefix = self.data_prefix.get("img", "")
                    if img_prefix and not img_path.startswith(img_prefix):
                        img_info["img_path"] = osp.join(img_prefix, img_path)
            if self.default_cam_key is not None:
                info["img_path"] = info["images"][self.default_cam_key]["img_path"]
                info["depth2img"] = np.array(
                    info["images"][self.default_cam_key]["depth2img"], dtype=np.float32
                )

        if not self.test_mode:
            # used in traing
            info["ann_info"] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info["eval_ann_info"] = self.parse_ann_info(info)

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Override from Det3DDataset to properly convert gt_bboxes_3d
        to DepthInstance3DBoxes for 3D augmentation transforms.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`
        """
        ann_info = super().parse_ann_info(info)
        # process data without any annotations
        if ann_info is None:
            ann_info = dict()
            ann_info["gt_bboxes_3d"] = np.zeros((0, 6), dtype=np.float32)
            ann_info["gt_labels_3d"] = np.zeros((0,), dtype=np.int64)
        # to target box structure (DepthInstance3DBoxes)
        ann_info["gt_bboxes_3d"] = DepthInstance3DBoxes(
            ann_info["gt_bboxes_3d"], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        return ann_info

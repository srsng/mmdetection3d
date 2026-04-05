# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class TOSceneDataset(Det3DDataset):
    """TO-Scene Dataset for 3D Detection.

    TO-Scene is a large-scale 3D detection dataset derived from ScanNet,
    containing 70 classes (18 large objects + 52 small objects).

    Please refer to:
    - DSPDet3D: https://github.com/Jimmyqwq/DSPDet3D
    - TO-Scene: https://drive.google.com/file/d/12IVVEt5kUQrz0_Qis58TH6Fj4yr6cJTo/view

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for data. Defaults to
            dict(pts='points',
                 pts_instance_mask='instance_mask',
                 pts_semantic_mask='semantic_mask').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' (for indoor datasets).
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """

    METAINFO = {
        "classes": (
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
        ),
    }

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        metainfo: Optional[dict] = None,
        data_prefix: dict = dict(
            pts="points",
            pts_instance_mask="instance_mask",
            pts_semantic_mask="semantic_mask",
        ),
        pipeline: List[Union[dict, Callable]] = [],
        modality: dict = dict(use_camera=False, use_lidar=True),
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
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs,
        )

        # assert "use_camera" in self.modality and "use_lidar" in self.modality
        # assert self.modality["use_camera"] or self.modality["use_lidar"]

    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): Info of a single sample data.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if "axis_align_matrix" in info:
            return np.array(info["axis_align_matrix"])
        else:
            warnings.warn(
                "axis_align_matrix is not found in TO-Scene data info, "
                "using identity matrix instead."
            )
            return np.eye(4).astype(np.float32)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `axis_align_matrix'.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info["axis_align_matrix"] = self._get_axis_align_matrix(info)
        info["pts_instance_mask_path"] = osp.join(
            self.data_prefix.get("pts_instance_mask", ""),
            info["pts_instance_mask_path"],
        )
        info["pts_semantic_mask_path"] = osp.join(
            self.data_prefix.get("pts_semantic_mask", ""),
            info["pts_semantic_mask_path"],
        )

        info = super().parse_data_info(info)
        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`.
        """
        ann_info = super().parse_ann_info(info)
        # empty gt
        if ann_info is None:
            ann_info = dict()
            ann_info["gt_bboxes_3d"] = np.zeros((0, 6), dtype=np.float32)
            ann_info["gt_labels_3d"] = np.zeros((0,), dtype=np.int64)
        # to target box structure

        ann_info["gt_bboxes_3d"] = DepthInstance3DBoxes(
            ann_info["gt_bboxes_3d"],
            box_dim=ann_info["gt_bboxes_3d"].shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5),
        ).convert_to(self.box_mode_3d)

        return ann_info

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
        'classes': ('keyboard', 'laptop', 'book', 'cup', 'mug',
                    'pen', 'notebook', 'phone'),
        'palette': [
            (230, 25, 72),    # keyboard - red
            (60, 180, 75),    # laptop - green
            (255, 225, 25),   # book - yellow
            (0, 130, 200),    # cup - blue
            (245, 130, 48),   # mug - orange
            (145, 30, 180),   # pen - purple
            (70, 240, 240),   # notebook - cyan
            (240, 50, 230),   # phone - magenta
        ]
    }

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
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)
        # to target box structure (DepthInstance3DBoxes)
        ann_info['gt_bboxes_3d'] = DepthInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        return ann_info

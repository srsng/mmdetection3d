# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from mmdet3d.models.layers import PointFPModule, build_sa_module
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig


@MODELS.register_module()
class DensityAwarePointNet2(nn.Module):
    """Density-aware PointNet2 backbone with SA and FP modules.

    This backbone integrates density information into SA layers for
    density-guided feature extraction and sampling. The density channel
    is appended to input points via ComputePointDensity transform.

    The key difference from standard PointNet2 is that this module:
    1. Applies density weighting to features in SA layers (except the last SA)
    2. Preserves density information for upsampling in FP layers

    Args:
        in_channels (int): Input channels of point cloud.
            Defaults to 4 (xyz + density).
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radius (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        fp_channels (tuple[tuple[int]]): Out channels of each mlp in FP module.
        norm_cfg (dict): Config of normalization layer.
            Defaults to dict(type='BN2d').
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
        init_cfg (dict, optional): Initialization config.

    Example:
        >>> backbone = DensityAwarePointNet2(
        ...     in_channels=4,
        ...     num_points=(2048, 1024, 512, 256),
        ...     radius=(0.2, 0.4, 0.8, 1.2),
        ...     num_samples=(64, 32, 16, 16),
        ...     sa_channels=((64, 64, 128), (128, 128, 256),
        ...                    (128, 128, 256), (128, 128, 256)),
        ...     fp_channels=((256, 256), (256, 256)),
        ...     sa_cfg=dict(type='PointSAModule', pool_mod='max',
        ...                   use_xyz=True, normalize_xyz=True))
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_points: Sequence[int] = (2048, 1024, 512, 256),
        radius: Sequence[float] = (0.2, 0.4, 0.8, 1.2),
        num_samples: Sequence[int] = (64, 32, 16, 16),
        sa_channels: Sequence[Sequence[int]] = ((64, 64, 128),
                                                  (128, 128, 256),
                                                  (128, 128, 256),
                                                  (128, 128, 256)),
        fp_channels: Sequence[Sequence[int]] = ((256, 256), (256, 256)),
        norm_cfg: ConfigType = dict(type='BN2d'),
        sa_cfg: ConfigType = dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True),
        init_cfg: OptMultiConfig = None
    ) -> None:
        super().__init__()
        self.num_sa = len(sa_channels)
        self.num_fp = len(fp_channels)

        assert len(num_points) == len(radius) == len(num_samples) == len(
            sa_channels)
        assert len(sa_channels) >= len(fp_channels)

        self.num_points = num_points
        self.radius = radius
        self.num_samples = num_samples

        # SA modules
        self.SA_modules = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
            sa_out_channel = cur_sa_mlps[-1]

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radius=radius[sa_index],
                    num_sample=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg))
            skip_channel_list.append(sa_out_channel)
            # For subsequent SA layers, we need to account for density channel
            # that is added after each SA layer (except the last)
            if sa_index < self.num_sa - 1:
                sa_in_channel = sa_out_channel

        # FP modules
        self.FP_modules = nn.ModuleList()

        fp_source_channel = skip_channel_list.pop()
        fp_target_channel = skip_channel_list.pop()
        for fp_index in range(len(fp_channels)):
            cur_fp_mlps = list(fp_channels[fp_index])
            cur_fp_mlps = [fp_source_channel + fp_target_channel] + cur_fp_mlps
            self.FP_modules.append(
                PointFPModule(mlp_channels=cur_fp_mlps))
            if fp_index != len(fp_channels) - 1:
                fp_source_channel = cur_fp_mlps[-1]
                fp_target_channel = skip_channel_list.pop()

    def _split_point_feats(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """Split point features into xyz and feature channels.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            tuple[torch.Tensor]: xyz and features.
        """
        xyz = points[..., :3].contiguous()
        features = points[..., 3:].transpose(1, 2).contiguous()  # (B, C, N)
        return xyz, features

    def forward(self, points: Tensor) -> Dict[str, List[Tensor]]:
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).
                The last feature channel should be density.

        Returns:
            dict[str, list[torch.Tensor]]: Outputs after SA and FP modules.
        """
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        # SA layers with density weighting
        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])

            # Apply density weighting for all SA layers except the last one
            # This preserves raw features in the last SA for upsampling
            if i < self.num_sa - 1:
                # cur_features: (B, C, N), last channel is density
                density = cur_features[:, -1:, :]  # (B, 1, N)
                cur_features_before_density = cur_features[:, :-1, :]  # (B, C-1, N)
                # Apply density weighting: enhance sparse regions
                cur_features = cur_features_before_density * density
                # Append density channel back for next SA layer
                cur_features = torch.cat([cur_features, density], dim=1)

            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))

        # FP layers
        fp_xyz = [sa_xyz[-1]]
        fp_features = [sa_features[-1]]
        fp_indices = [sa_indices[-1]]

        for i in range(self.num_fp):
            fp_features.append(self.FP_modules[i](
                sa_xyz[self.num_sa - i - 1], sa_xyz[self.num_sa - i],
                sa_features[self.num_sa - i - 1], fp_features[-1]))
            fp_xyz.append(sa_xyz[self.num_sa - i - 1])
            fp_indices.append(sa_indices[self.num_sa - i - 1])

        ret = dict(
            fp_xyz=fp_xyz,
            fp_features=fp_features,
            fp_indices=fp_indices,
            sa_xyz=sa_xyz,
            sa_features=sa_features,
            sa_indices=sa_indices)
        return ret

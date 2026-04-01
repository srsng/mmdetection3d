"""CGNL (Compact Generalized Non-Local) Neck 模块。

用于增强 3D 检测网络的局部特征关联。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class CGNLBlock(nn.Module):
    """Compact Generalized Non-Local Block for channel-wise local feature modeling.

    This block applies non-local attention with channel grouping and layer normalization
    to model spatial correlations in point cloud features.

    Args:
        in_channels (int): Number of input channels.
        groups (int): Number of channel groups for grouped convolution.
            Defaults to 4.
        reduction (int): Channel reduction ratio. Defaults to 4.
        use_scale (bool): Whether to use scale factor in attention.
            Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        groups: int = 4,
        reduction: int = 4,
        use_scale: bool = True,
    ):
        super().__init__()
        assert in_channels % groups == 0, (
            f'in_channels ({in_channels}) must be divisible by groups ({groups})'
        )
        self.in_channels = in_channels
        self.groups = groups
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        assert self.inter_channels % groups == 0, (
            f'inter_channels ({self.inter_channels}) must be divisible by groups ({groups})'
        )

        # Grouped 1x1 convolutions for theta, phi, g
        self.theta = nn.Conv1d(
            in_channels, self.inter_channels, kernel_size=1, groups=groups)
        self.phi = nn.Conv1d(
            in_channels, self.inter_channels, kernel_size=1, groups=groups)
        self.g = nn.Conv1d(
            in_channels, self.inter_channels, kernel_size=1, groups=groups)

        # 1x1 convolution for output
        self.out_conv = nn.Conv1d(
            self.inter_channels, in_channels, kernel_size=1, groups=groups)

        # LayerNorm for residual connection
        self.ln = nn.LayerNorm(self.in_channels)

        # Initialize
        nn.init.constant_(self.out_conv.weight, 0)
        nn.init.constant_(self.out_conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input features with shape (B, C, N).

        Returns:
            torch.Tensor: Output features with shape (B, C, N).
        """
        batch_size, channels, num_points = x.size()

        # Compute theta, phi, g
        theta_x = self.theta(x)  # (B, C', N)
        phi_x = self.phi(x)  # (B, C', N)
        g_x = self.g(x)  # (B, C', N)

        # Compute attention weights: theta_phi = bmm(theta.permute(0,2,1), phi)
        theta_x = theta_x.permute(0, 2, 1)  # (B, N, C')
        theta_phi = torch.bmm(theta_x, phi_x)  # (B, N, N)

        if self.use_scale:
            theta_phi = theta_phi / (self.inter_channels**0.5)

        attention = F.softmax(theta_phi, dim=-1)

        # Apply attention to g
        g_x = g_x.permute(0, 2, 1)  # (B, N, C')
        out = torch.bmm(attention, g_x)  # (B, N, C')
        out = out.permute(0, 2, 1)  # (B, C', N)

        # Upscale and apply residual connection with LayerNorm
        out = self.out_conv(out)
        # LayerNorm on (B, N, C), then permute back to (B, C, N)
        out = out + self.ln(out.permute(0, 2, 1)).permute(0, 2, 1)

        return out


@MODELS.register_module()
class CGNLNeck(nn.Module):
    """CGNL Neck module for MMDetection3D.

    Args:
        in_channels (list[int]): List of input channels.
        out_channels (list[int]): List of output channels.
        num_blocks (int): Number of CGNL blocks per layer. Defaults to 1.
        groups (int): Number of channel groups. Defaults to 4.
        reduction (int): Channel reduction ratio. Defaults to 4.
        use_scale (bool): Whether to use scale factor. Defaults to True.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        num_blocks: int = 1,
        groups: int = 4,
        reduction: int = 4,
        use_scale: bool = True,
    ):
        super().__init__()
        assert len(in_channels) == len(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.groups = groups

        # Create CGNL blocks for each feature layer
        self.cgnl_blocks = nn.ModuleList()
        self.adapt_convs = nn.ModuleList()

        for in_ch, out_ch in zip(in_channels, out_channels):
            blocks = nn.ModuleList(
                [CGNLBlock(in_ch, groups=groups, reduction=reduction, use_scale=use_scale)
                 for _ in range(num_blocks)]
            )
            self.cgnl_blocks.append(blocks)

            # Channel adaptation conv
            if in_ch != out_ch:
                adapt_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)
            else:
                adapt_conv = nn.Identity()
            self.adapt_convs.append(adapt_conv)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        前向传播。

        Args:
            inputs: 输入特征列表，每个特征形状为 (B, C, N)

        Returns:
            输出特征列表
        """
        outputs = []

        for i, x in enumerate(inputs):
            # 应用 CGNL 块
            for block in self.cgnl_blocks[i]:
                x = block(x)

            # 通道适配
            x = self.adapt_convs[i](x)

            outputs.append(x)

        return outputs


@MODELS.register_module()
class VoteNetCGNLNeck(nn.Module):
    """CGNL Neck adapted for VoteNet backbone output.

    VoteNet's PointNet2SASSG backbone outputs a dictionary format:
        {
            'fp_xyz': [points_0, points_1, ...],
            'fp_features': [feats_0, feats_1, ...],
            'fp_indices': [indices_0, indices_1, ...]
        }

    According to the paper, CGNL should be applied after the last upsampling
    layer to enhance local feature correlation.

    Args:
        in_channels (int or list[int]): Input channels.
        out_channels (int or list[int]): Output channels.
        num_blocks (int): Number of CGNL blocks. Defaults to 1.
        groups (int): Number of channel groups. Defaults to 4.
        reduction (int): Channel reduction ratio. Defaults to 4.
        use_scale (bool): Whether to use scale factor. Defaults to True.
        target_layer_idx (int): Target feature layer index, -1 for last layer.
            Defaults to -1.
    """

    def __init__(
        self,
        in_channels: int | list[int],
        out_channels: int | list[int],
        num_blocks: int = 1,
        groups: int = 4,
        reduction: int = 4,
        use_scale: bool = True,
        target_layer_idx: int = -1,
    ):
        super().__init__()
        self.target_layer_idx = target_layer_idx
        self.groups = groups

        # Normalize to list format
        if isinstance(in_channels, int):
            self.in_channels = [in_channels]
            self.out_channels = (
                [out_channels] if isinstance(out_channels, int) else out_channels
            )
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels

        # Create CGNL blocks
        self.cgnl_blocks = nn.ModuleList()
        self.adapt_convs = nn.ModuleList()

        for in_ch, out_ch in zip(self.in_channels, self.out_channels):
            blocks = nn.ModuleList(
                [CGNLBlock(in_ch, groups=groups, reduction=reduction, use_scale=use_scale)
                 for _ in range(num_blocks)]
            )
            self.cgnl_blocks.append(blocks)

            if in_ch != out_ch:
                adapt_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)
            else:
                adapt_conv = nn.Identity()
            self.adapt_convs.append(adapt_conv)

    def forward(
        self, inputs: dict[str, list[torch.Tensor]] | list[torch.Tensor]
    ) -> dict[str, list[torch.Tensor]]:
        """
        前向传播。

        Args:
            inputs: VoteNet backbone 输出的字典，或标准特征列表

        Returns:
            处理后的字典（适配 VoteHead）
        """
        # 处理 VoteNet 字典格式输入
        if isinstance(inputs, dict):
            fp_xyz = inputs.get("fp_xyz", [])
            fp_features = inputs.get("fp_features", [])
            fp_indices = inputs.get("fp_indices", [])
        else:
            # 如果是列表，假设它是 fp_features
            fp_xyz = []
            fp_features = inputs
            fp_indices = []

        if not fp_features:
            raise ValueError("No features found in input")

        # 确定目标层
        target_idx = (
            self.target_layer_idx
            if self.target_layer_idx >= 0
            else len(fp_features) - 1
        )

        outputs = []
        for i, x in enumerate(fp_features):
            if i == target_idx:
                # 对目标层应用 CGNL
                for block in self.cgnl_blocks[0]:  # 只用一个 CGNL 块
                    x = block(x)
                x = self.adapt_convs[0](x)
            outputs.append(x)

        # 返回字典格式，适配 VoteHead
        return {"fp_xyz": fp_xyz, "fp_features": outputs, "fp_indices": fp_indices}


@MODELS.register_module()
class CGNLLocalFusionNeck(nn.Module):
    """CGNL Local Fusion Neck for point cloud feature enhancement.

    This neck takes point cloud data with xyz + features (e.g., density-aware
    features from DensityAwarePointNet2), applies CGNL attention to the feature
    channels, and outputs in a format compatible with VoteHead.

    The input format is (B, N, 3+D) where:
        B: batch size
        N: number of points
        3: xyz coordinates
        D: additional feature channels (including density)

    Args:
        in_channels (int): Number of input feature channels (excluding xyz).
            Should match the feature dimension from backbone.
        num_blocks (int): Number of CGNL blocks. Defaults to 1.
        groups (int): Number of channel groups for CGNLBlock. Defaults to 1.
            Set to 1 for flexible channel support.
        reduction (int): Channel reduction ratio. Defaults to 4.
        use_scale (bool): Whether to use scale factor. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        num_blocks: int = 1,
        groups: int = 4,
        reduction: int = 4,
        use_scale: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.groups = groups

        # Create CGNL blocks for feature processing
        self.cgnl_blocks = nn.ModuleList([
            CGNLBlock(
                in_channels=in_channels,
                groups=groups,
                reduction=reduction,
                use_scale=use_scale,
            ) for _ in range(num_blocks)
        ])

    def forward(
        self,
        inputs: dict[str, list[torch.Tensor]] | torch.Tensor
    ) -> dict:
        """Forward pass.

        Args:
            inputs: Either:
                - VoteNet backbone输出的字典，格式为
                  {"fp_xyz": [...], "fp_features": [...], "fp_indices": [...]}
                - 或者tensor格式 (B, N, 3 + in_channels)

        Returns:
            dict: Output dictionary with keys:
                - fp_xyz: list of xyz coordinates (one layer)
                - fp_features: list of processed features (one layer)
                - fp_indices: list of indices (one layer, identity)
        """
        # 处理 VoteNet 字典格式输入 (来自 backbone)
        if isinstance(inputs, dict):
            fp_xyz = inputs.get("fp_xyz", [])
            fp_features = inputs.get("fp_features", [])
            fp_indices = inputs.get("fp_indices", [])

            if not fp_features:
                raise ValueError("No features found in input")

            # 只对最后一层应用 CGNL
            target_idx = len(fp_features) - 1
            x = fp_features[target_idx]  # (B, C, N)

            # 应用 CGNL blocks
            for block in self.cgnl_blocks:
                x = block(x)

            # 更新 features
            fp_features = list(fp_features)
            fp_features[target_idx] = x

            return {
                "fp_xyz": fp_xyz,
                "fp_features": fp_features,
                "fp_indices": fp_indices,
            }

        # 处理直接 tensor 输入 (B, N, 3 + in_channels)
        points = inputs
        batch_size, num_points = points.shape[:2]

        # Separate xyz and features
        xyz = points[..., :3]  # (B, N, 3)
        features = points[..., 3:]  # (B, N, in_channels)

        # Transpose for Conv1d: (B, N, C) -> (B, C, N)
        features = features.permute(0, 2, 1).contiguous()  # (B, C, N)

        # Apply CGNL blocks
        for block in self.cgnl_blocks:
            features = block(features)

        # Keep (B, C, N) format for VoteHead compatibility
        # Note: VoteHead._extract_input expects seed_features in (B, C, N) format

        # Create identity indices for compatibility
        indices = torch.arange(
            num_points, device=points.device).unsqueeze(0).expand(
                batch_size, -1).long()

        # Return in VoteHead-compatible format
        # fp_features in (B, C, N) format for VoteModule
        return {
            "fp_xyz": [xyz],
            "fp_features": [features],
            "fp_indices": [indices],
        }

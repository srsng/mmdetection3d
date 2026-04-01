# 基于OpenMMLab MMDetection3D 实现密度-局部特征融合3D目标检测网络 任务指导书

## 一、项目概述

### 1.1 项目背景
本项目基于论文《Point cloud 3D object detection method based on density information‑local feature fusion》，针对现有VoteNet等点云3D检测算法**点云密度不均导致稀疏区域信息丢失、局部特征关联不足**的核心痛点，在OpenMMLab MMDetection3D（简称mmdet3d）框架下，完整复现论文提出的**密度信息-局部特征融合3D目标检测网络**，实现室内场景点云3D目标检测的精度与鲁棒性提升。

### 1.2 核心目标
1. 完整实现论文的两个核心创新模块：**密度感知特征提取模块**、**CGNL局部特征融合模块**；
2. 在SUN RGB-D数据集上完成模型训练与验证；
3. 支持后续扩展到mini版SUN RGB-D数据集。

### 1.3 项目范围
- 适配数据集：第一阶段完整SUN RGB-D，第二阶段支持mini SUN RGB-D
- 基线模型：mmdet3d官方实现的VoteNet
- 核心交付：可运行的网络代码、配置文件、训练/验证脚本
- 不包含：消融实验、鲁棒性实验、可视化报告、ScanNetV2数据集

### 1.4 参考资料
1.  目标论文：Point cloud 3D object detection method based on density information‑local feature fusion
2.  mmdet3d官方文档：https://mmdetection3d.readthedocs.io/
3.  VoteNet官方实现与mmdet3d基线代码
4.  CGNL（Compact Generalized Non-Local Network）原论文与基础实现

## 二、前置准备

### 2.1 环境状态
**已配置完成**

如需确认版本，执行：
```bash
pip list | grep -E "torch|mmdet|mmcv|mmengine"
```

### 2.2 数据集状态
**数据集已准备完毕，完全可用，无需额外处理。**

#### 目录结构
```
sunrgbd/
├── README.md
├── sunrgbd.md
├── sunrgbd_trainval/
│   ├── calib/          # .txt 相机标定文件
│   ├── depth/          # .mat 点云文件，包含 xyz 坐标和 rgb 色彩值
│   ├── image/          # .jpg 二维图像文件
│   ├── label/          # .txt 检测任务标注数据（版本二）
│   ├── label_v1/       # .txt 检测任务标注数据（版本一）
│   ├── seg_label/      # .txt 分割任务标注数据
│   ├── train_data_idx.txt
│   └── val_data_idx.txt
├── points/             # 降采样后的点云数据 .bin 文件
├── sunrgbd_infos_train.pkl
└── sunrgbd_infos_val.pkl
```

#### 数据集规模
- 训练集：5285 个样本
- 验证集：5050 个样本

#### sunrgbd_infos_xxx.pkl 数据结构
每个 info 包含：
- `lidar_points`: `{'num_pts_feats': 点数特征维度, 'lidar_path': 点云文件名}`
- `images`: `{'CAM0': {'img_path', 'depth2img': (4,4), 'height', 'width'}}`
- `instances`: `[{'bbox_3d': [7], 'bbox': [4], 'bbox_label_3d': int, 'bbox_label': int}, ...]`

**mini SUN RGB-D 数据集结构与完整版相同。**

### 2.3 前置基线验证（可选）
如需验证环境与数据集，可跑通mmdet3d官方VoteNet基线模型：
1.  跑通mmdet3d官方VoteNet基线模型，确保环境、数据集无问题；
2.  熟悉mmdet3d核心逻辑：注册器机制、配置文件体系、PointNet++骨干、VoteNet检测头、数据Pipeline流程。

## 三、硬件与训练策略

### 3.1 硬件说明
| 场景 | GPU | 显存 | 用途 |
|------|-----|------|------|
| 调试 | RTX 3060 | 6GB | 代码调试、小批量验证 |
| 正式训练 | RTX 4070 | 12GB | 模型训练 |

### 3.2 训练策略
- **调试阶段**：batch_size=1~2，验证代码逻辑正确
- **正式训练**：batch_size=4（4070可支持），epoch数待定
- **两阶段微调**（可选）：
  1. 冻结VoteNet预训练权重，仅训练新增模块
  2. 解冻全网络，端到端微调

## 四、核心任务拆解与实施步骤

### 任务1：核心模块1——密度信息特征提取模块实现
**交付物**：密度计算Transform、密度感知PointNet++骨干代码

#### 1.1 点云密度计算模块
基于高斯核的KDE（核密度估计）密度计算：
1.  用高斯核函数计算每个点的空间密度；
2.  计算逆密度，降低稠密区域权重、增强稀疏区域权重；
3.  用k近邻（k=64）优化KDE计算效率。

**代码路径**：`mmdet3d/datasets/transforms/transforms_3d.py`

```python
@TRANSFORMS.register_module()
class ComputePointDensity:
    """计算点云密度的Transform，与论文KDE密度计算对齐"""
    def __init__(self, kernel='gaussian', sigma=0.1, k_neighbor=64, epsilon=1e-8):
        self.kernel = kernel
        self.sigma = sigma
        self.k_neighbor = k_neighbor
        self.epsilon = epsilon

    def __call__(self, results):
        points = results['points'].tensor[:, :3].numpy()
        tree = BallTree(points, leaf_size=40)
        distances, _ = tree.query(points, k=self.k_neighbor)

        if self.kernel == 'gaussian':
            kernel_vals = np.exp(-(distances ** 2) / (2 * self.sigma ** 2)) / (np.sqrt(2 * np.pi) * self.sigma)
            density = np.sum(kernel_vals, axis=1) / self.k_neighbor
        else:
            raise NotImplementedError

        inv_density = 1 / (density + self.epsilon)
        inv_density = (inv_density - inv_density.min()) / (inv_density.max() - inv_density.min() + self.epsilon)

        results['point_density'] = inv_density.astype(np.float32)
        results['points'].tensor = torch.cat([results['points'].tensor, torch.from_numpy(inv_density).unsqueeze(1)], dim=-1)
        return results
```

#### 1.2 密度感知PointNet++骨干
**4层下采样（SA）+2层上采样（FP）**结构，在每一层SA层中融入密度信息。

**代码路径**：`mmdet3d/models/backbones/density_aware_pointnet2.py`

```python
@BACKBONES.register_module()
class DensityAwarePointNet2(nn.Module):
    def __init__(self, in_channels=4, sa_cfg=None, fp_cfg=None):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        for sa_param in sa_cfg:
            self.SA_modules.append(PointNet2SAModule(**sa_param))
        self.FP_modules = nn.ModuleList()
        for fp_param in fp_cfg:
            self.FP_modules.append(PointNet2FPModule(**fp_param))

    def forward(self, points):
        xyz = points[..., :3].contiguous()
        features = points[..., 3:].transpose(1, 2).contiguous()
        l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            if i < len(self.SA_modules)-1:
                density = li_features[:, -1:, :]
                li_features = li_features[:, :-1, :] * density
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules)+1), -1):
            l_features[i-1] = self.FP_modules[i](l_xyz[i-1], l_xyz[i], l_features[i-1], l_features[i])

        return torch.cat([l_xyz[0], l_features[0].transpose(1, 2)], dim=-1)
```

注册：在`mmdet3d/models/backbones/__init__.py`中导入并注册。

### 任务2：核心模块2——CGNL局部特征融合模块实现
**交付物**：CGNL注意力模块、特征融合Neck代码

**代码路径**：`mmdet3d/models/necks/cgnl_neck.py`

```python
class CGNLBlock(nn.Module):
    def __init__(self, in_channels, groups=4, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.inter_channels = in_channels // reduction
        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.out_conv = nn.Conv1d(self.inter_channels, in_channels, kernel_size=1)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, N = x.shape
        residual = x
        x_group = x.reshape(B * self.groups, C // self.groups, N)
        theta = self.theta(x_group).reshape(B * self.groups, -1, N)
        phi = self.phi(x_group).reshape(B * self.groups, -1, N)
        g = self.g(x_group).reshape(B * self.groups, -1, N)
        theta_phi = torch.bmm(theta.permute(0, 2, 1), phi)
        attention = F.softmax(theta_phi, dim=-1)
        out = torch.bmm(g, attention.permute(0, 2, 1))
        out = self.out_conv(out.reshape(B, self.inter_channels, N))
        out = self.norm((out + residual).permute(0, 2, 1)).permute(0, 2, 1)
        return out

@NECKS.register_module()
class CGNLLocalFusionNeck(nn.Module):
    def __init__(self, in_channels, num_blocks=1, groups=4):
        super().__init__()
        self.blocks = nn.ModuleList([CGNLBlock(in_channels, groups=groups) for _ in range(num_blocks)])

    def forward(self, x):
        xyz = x[..., :3]
        features = x[..., 3:].transpose(1, 2)
        for block in self.blocks:
            features = block(features)
        return torch.cat([xyz, features.transpose(1, 2)], dim=-1)
```

注册：在`mmdet3d/models/necks/__init__.py`中导入并注册。

### 任务3：模型整体搭建与检测头适配
**交付物**：完整模型注册

```
输入点云(N×3) → 密度计算 → 密度感知PointNet2骨干 → CGNL融合Neck → VoteNet投票头 → 提案生成 → 3D NMS → 输出3D框
```

1.  沿用mmdet3d官方`VoteHead`，仅适配输入特征维度；
2.  完成模型整体注册，确保配置文件可直接调用。

### 任务4：数据Pipeline与配置文件编写
**交付物**：SUN RGB-D完整配置文件

**Pipeline修改**：
```python
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', load_dim=6, use_dim=[0,1,2]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointSample', num_points=10000),
    dict(type='ComputePointDensity', sigma=0.1, k_neighbor=64),
    # 其余数据增强沿用VoteNet配置
]
```

**配置文件**：`configs/density_votenet/density_votenet_8xb4-sunrgbd-3d.py`

```python
_base_ = [
    '../_base_/datasets/sunrgbd-3d.py',
    '../_base_/models/votenet.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='VoteNet',
    backbone=dict(
        type='DensityAwarePointNet2',
        in_channels=4,
        sa_cfg=[
            dict(num_point=2048, radius=0.2, num_sample=64, mlp_channels=[4, 64, 64, 128]),
            dict(num_point=1024, radius=0.4, num_sample=64, mlp_channels=[128+1, 128, 128, 256]),
            dict(num_point=512, radius=0.8, num_sample=64, mlp_channels=[256+1, 128, 128, 256]),
            dict(num_point=256, radius=1.2, num_sample=64, mlp_channels=[256+1, 128, 128, 256]),
        ],
        fp_cfg=[
            dict(mlp_channels=[512, 256, 256]),
            dict(mlp_channels=[384, 256, 256]),
        ]),
    neck=dict(
        type='CGNLLocalFusionNeck',
        in_channels=256,
        num_blocks=1,
        groups=4),
    bbox_head=dict(type='VoteHead', vote_module_cfg=dict(in_channels=256, ...))
)
```

### 任务5：训练与指标可视化
**交付物**：训练完成的模型权重、mAP指标

1.  **调试阶段**（3060 6GB）：
    ```bash
    python tools/train.py configs/density_votenet/density_votenet_8xb4-sunrgbd-3d.py --cfg-options train_dataloader.batch_size=1
    ```
2.  **正式训练**（4070 12GB）：
    ```bash
    python tools/train.py configs/density_votenet/density_votenet_8xb4-sunrgbd-3d.py
    ```
3.  **验证与指标**：
    ```bash
    python tools/test.py configs/density_votenet/density_votenet_8xb4-sunrgbd-3d.py --checkpoint work_dirs/xxx.pth
    ```
4.  **训练指标可视化**：mmdet3d支持TensorBoard/WandB，记录训练过程的mAP、loss等指标。

### 任务6：mini数据集支持（第二阶段）
**交付物**：mini SUN RGB-D数据集配置文件

1.  实现mini版SUN RGB-D数据集转换器
2.  编写mini版训练配置文件
3.  验证mini数据集上模型能正常训练

## 五、关键技术难点与解决方案

| 序号 | 技术难点 | 解决方案 |
| ---- | ------- | -------- |
| 1 | KDE密度计算效率低 | 用BallTree加速k近邻搜索（k=64），降低计算量 |
| 2 | 密度加权导致数值不稳定 | 对逆密度做min-max归一化，加入epsilon防止除零；保留残差连接 |
| 3 | CGNL模块显存占用高 | 通道分组（4组）+ reduction=4压缩中间通道 |
| 4 | mmdet3d Pipeline密度传递 | 在PointSample之后计算密度，拼接到点云张量额外通道 |

## 六、项目验收标准

### 6.1 功能性验收
1.  代码符合mmdet3d开发规范，所有模块完成注册；
2.  支持训练、验证流程，无报错；
3.  密度模块、CGNL模块正常工作。

### 6.2 性能验收
1.  SUN RGB-D验证集mAP指标正常（具体数值待训练后确认）；
2.  模型收敛，loss正常下降。

### 6.3 文档验收
1.  代码注释完整；
2.  运行步骤文档清晰。

## 七、项目排期

| 周期 | 核心任务 | 交付物 |
| ---- | -------- | ------ |
| Day1 | VoteNet基线验证（可选） | 基线可运行 |
| Day2-3 | 密度计算模块实现 | ComputePointDensity代码 |
| Day4-5 | 密度感知PointNet++骨干改造 | DensityAwarePointNet2代码 |
| Day6-7 | CGNL融合模块实现 | CGNLLocalFusionNeck代码 |
| Day8 | 模型整体搭建与注册 | 完整模型代码 |
| Day9-10 | 数据Pipeline适配、配置文件编写 | 训练配置文件 |
| Day11-12 | 调试阶段训练（3060，小batch） | 验证代码正确性 |
| Day13+ | 正式训练（4070） | 模型权重 |

注：数据集准备已提前完成，故排期从Day2开始。

## 八、风险与应急预案

| 风险类型 | 风险描述 | 应急预案 |
| -------- | -------- | -------- |
| 显存不足 | 3060调试时batch_size=1仍OOM | 降低输入点云数量（如从10000降至5000） |
| 指标风险 | 训练后mAP低于预期 | 核对论文超参数，逐模块单元测试验证 |
| 进度风险 | 核心模块开发延期 | 优先实现核心逻辑，先跑通再优化 |
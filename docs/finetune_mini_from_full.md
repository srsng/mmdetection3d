# VoteNet 微调指南: 从完整SUNRGBD到Mini SUNRGBD

## 概述

本文档说明如何使用在完整SUNRGBD数据集(10类)上训练的VoteNet模型，在Mini SUNRGBD数据集(5类)上进行微调。

## 数据集差异

| 属性 | 完整SUNRGBD | Mini SUNRGBD |
|------|------------|--------------|
| 类别数 | 10 | 5 |
| 类别 | bed, table, sofa, chair, toilet, desk, dresser, night_stand, bookshelf, bathtub | keyboard, laptop, book, cup, mug |
| 数据路径 | data2/sunrgbd/ | data2/mini_sunrgbd/ |

## 关键问题

由于完整数据集和mini数据集的**类别数不同**，`bbox_head`的输出层维度不同(10 vs 5)，因此：
- **不能直接加载完整数据集的完整checkpoint**
- 需要**跳过`bbox_head`权重**，只加载`backbone`和`neck`的预训练权重

## 微调流程

### Step 1: 训练完整数据集模型

```bash
cd /path/to/mmdetection3d
python tools/train.py configs/density_votenet/density_votenet_8xb4-sunrgbd-3d.py
```

训练完成后，checkpoint保存在:
```
work_dirs/density_votenet_8xb4-sunrgbd-3d/epoch_XX.pth
```

### Step 2: 提取backbone和neck权重

使用提供的脚本提取权重(跳过bbox_head):

```bash
python tools/extract_backbone_neck.py \
    --input work_dirs/density_votenet_8xb4-sunrgbd-3d/epoch_20.pth \
    --output work_dirs/density_votenet_8xb4-sunrgbd-3d/backbone_neck_finetune.pth
```

或使用dry-run模式查看将要提取的内容:

```bash
python tools/extract_backbone_neck.py \
    --input work_dirs/density_votenet_8xb4-sunrgbd-3d/epoch_20.pth \
    --dry-run
```

### Step 3: 在Mini数据集上微调

确保Mini数据集存在于 `data2/mini_sunrgbd/`

```bash
cd /path/to/4070-machine  # 切换到有足够显存的机器

python tools/train.py \
    configs/density_votenet/density_votenet_mini_sunrgbd_finetune.py
```

训练输出将保存在:
```
work_dirs/density_votenet_mini_sunrgbd_finetune/
```

### Step 4: 监控训练

启动TensorBoard查看训练曲线:

```bash
tensorboard --logdir work_dirs/density_votenet_mini_sunrgbd_finetune/
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `tools/extract_backbone_neck.py` | 权重裁剪脚本 |
| `configs/density_votenet/density_votenet_mini_sunrgbd_finetune.py` | 微调配置文件 |
| `configs/density_votenet/density_votenet_mini_sunrgbd.py` | Mini数据集基础配置(5类) |
| `configs/density_votenet/density_votenet_8xb4-sunrgbd-3d.py` | 完整数据集配置(10类) |

## 训练配置说明

`density_votenet_mini_sunrgbd_finetune.py` 的关键配置:

```python
# 加载预训练的backbone和neck权重
load_from = '../density_votenet_8xb4-sunrgbd-3d/backbone_neck_finetune.pth'

# 继承自mini数据集基础配置(5类)
_base_ = ['./density_votenet_mini_sunrgbd.py']

# 使用较小的学习率进行微调
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, end=1),
    dict(type='MultiStepLR', begin=1, end=20, milestones=[15, 18], gamma=0.1)
]
```

## 注意事项

1. **学习率**: 微调时通常使用比从头训练更小的学习率
2. **Epoch数**: 由于是从预训练模型开始，20 epoch通常足够
3. **batch_size**: Mini数据集较小，batch_size=4应该可以正常工作
4. **显存**: RTX 4070 12GB 足够运行完整训练

## 常见问题

**Q: 为什么bbox_head不能加载预训练权重?**
A: 因为完整数据集有10类，mini数据集只有5类，分类层的输出维度不同。

**Q: backbone和neck的权重为什么可以复用?**
A: 这两层负责特征提取，与具体类别数无关，因此可以复用。

**Q: 如何选择要加载的epoch checkpoint?**
A: 建议选择验证集mAP最高的checkpoint，通常不是最后一个epoch。

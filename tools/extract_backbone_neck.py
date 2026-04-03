#!/usr/bin/env python3
"""
从完整SUNRGBD数据集训练的VoteNet checkpoint中提取backbone和neck权重，
用于在mini SUNRGBD数据集上进行微调。

用法:
    python tools/extract_backbone_neck.py \
        --input work_dirs/density_votenet_8xb4-sunrgbd-3d/epoch_20.pth \
        --output work_dirs/density_votenet_8xb4-sunrgbd-3d/backbone_neck_finetune.pth

原理:
    完整数据集(10类)训练的checkpoint包含:
        - backbone.* (DensityAwarePointNet2权重, 兼容)
        - neck.* (CGNLLocalFusionNeck权重, 兼容)
        - bbox_head.* (分类头权重, 不兼容!)

    mini数据集只训练5类, bbox_head输出维度不同, 因此需要跳过该部分权重。
"""

import argparse
import torch


def extract_backbone_neck(input_path: str, output_path: str):
    """从checkpoint中提取backbone和neck权重, 跳过bbox_head"""

    print(f"Loading checkpoint from: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu')

    if 'state_dict' not in ckpt:
        raise ValueError(f"Invalid checkpoint format. Expected 'state_dict' key.")

    state_dict = ckpt['state_dict']

    # 过滤掉bbox_head相关的权重
    new_state_dict = {}
    skipped_keys = []

    for key, value in state_dict.items():
        if key.startswith('bbox_head.'):
            skipped_keys.append(key)
        else:
            new_state_dict[key] = value

    # 保存裁剪后的权重
    new_ckpt = {'state_dict': new_state_dict}

    print(f"Saving extracted weights to: {output_path}")
    torch.save(new_ckpt, output_path)

    print(f"\nExtraction summary:")
    print(f"  - Kept keys: {len(new_state_dict)}")
    print(f"  - Skipped keys (bbox_head): {len(skipped_keys)}")

    if skipped_keys:
        print(f"\nSkipped keys (first 10):")
        for key in skipped_keys[:10]:
            print(f"    - {key}")
        if len(skipped_keys) > 10:
            print(f"    ... and {len(skipped_keys) - 10} more")

    print(f"\nDone! Extracted weights saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract backbone and neck weights from VoteNet checkpoint')
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input checkpoint path')
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output checkpoint path')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be extracted without saving')

    args = parser.parse_args()

    if args.dry_run:
        print(f"[Dry run] Would extract from: {args.input}")
        ckpt = torch.load(args.input, map_location='cpu')
        state_dict = ckpt.get('state_dict', {})

        backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.')]
        neck_keys = [k for k in state_dict.keys() if k.startswith('neck.')]
        bbox_head_keys = [k for k in state_dict.keys() if k.startswith('bbox_head.')]

        print(f"  - backbone.* keys: {len(backbone_keys)}")
        print(f"  - neck.* keys: {len(neck_keys)}")
        print(f"  - bbox_head.* keys: {len(bbox_head_keys)} (would be skipped)")
        return

    extract_backbone_neck(args.input, args.output)


if __name__ == '__main__':
    main()

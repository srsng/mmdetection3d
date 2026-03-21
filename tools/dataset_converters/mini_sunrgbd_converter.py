# Copyright (c) OpenMMLab. All rights reserved.
"""Convert SUNRGBD to MiniSUNRGBD dataset for desktop objects detection.

MiniSUNRGBD is a subset of SUNRGBD containing 8 desktop object classes:
keyboard, laptop, book, cup, mug, pen, notebook, phone

This converter:
1. Loads MiniSUNRGBD.json to get target classes and sample IDs
2. Re-indexes classes to 0-7 for MiniSUNRGBD
3. Adds ~10% negative samples (samples without target classes)
4. Generates pkl with new label indices
"""

import argparse
import json
import os
import os.path as osp
import pickle
import random
from collections import defaultdict

import mmengine
import numpy as np


# MiniSUNRGBD class order (must match dataset METAINFO)
MINI_SUNRGBD_CLASSES = ('keyboard', 'laptop', 'book', 'cup', 'mug',
                         'pen', 'notebook', 'phone')
MINI_CLASS_TO_LABEL = {c: i for i, c in enumerate(MINI_SUNRGBD_CLASSES)}


def parse_sunrgbd_label_line(line: str) -> dict:
    """Parse a single line from SUNRGBD label file.

    Label format:
    classname xmin ymin xmax ymax centroid_x centroid_y centroid_z \
        width length height orientation_x orientation_y

    Returns:
        dict with parsed fields, or None if parsing fails
    """
    parts = line.strip().split()
    if len(parts) < 13:
        return None

    try:
        data = [float(x) for x in parts[1:]]
    except ValueError:
        return None

    # heading_angle = arctan2(orientation_y, orientation_x) = arctan2(data[11], data[10])
    # but some lines may have different format, use safe access
    if len(data) >= 12:
        heading_angle = np.arctan2(data[11], data[10])
    elif len(data) >= 11:
        heading_angle = data[10]
    else:
        heading_angle = 0.0

    return {
        'classname': parts[0],
        'bbox': np.array([data[0], data[1], data[0] + data[3], data[2] + data[4]]),  # x1, y1, x2, y2
        'center': np.array([data[5], data[6], data[7]]),  # cx, cy, cz
        'size': np.array([data[9], data[8], data[10]]) * 2,  # l, w, h (x_size, y_size, z_size)
        'heading_angle': heading_angle,
    }


def create_instance_dict(parsed: dict, label: int, classname: str) -> dict:
    """Create instance dict with 3D bbox in [cx, cy, cz, l, w, h, angle] format."""
    box3d = np.concatenate([
        parsed['center'],      # cx, cy, cz
        parsed['size'],        # l, w, h
        np.array([parsed['heading_angle']])  # angle
    ])
    return {
        'bbox': parsed['bbox'],          # 2D bbox [x1, y1, x2, y2]
        'bbox_label': label,             # 2D label (same as 3D for our purpose)
        'bbox_3d': box3d.astype(np.float32),  # 3D bbox [cx, cy, cz, l, w, h, angle]
        'bbox_label_3d': label,          # 3D label (MiniSUNRGBD index 0-7)
        'class_name': classname,         # original class name
    }


def filter_mini_sunrgbd_infos(
    root_path: str,
    mini_config_path: str,
    output_dir: str,
    sample_limit: int = 50,
    min_samples: int = 18,
    negative_ratio: float = 0.1,
) -> dict:
    """Filter MiniSUNRGBD subset from full SUNRGBD dataset.

    Args:
        root_path: Root path of original SUNRGBD data.
        mini_config_path: Path to MiniSUNRGBD.json config file.
        output_dir: Output directory for filtered pkl files.
        sample_limit: Maximum samples per class (default 50).
        min_samples: Minimum samples threshold to keep a class (default 18).
        negative_ratio: Ratio of negative samples to add (default 0.1 = 10%).

    Returns:
        Statistics dict with filtered dataset info.
    """
    # Load MiniSUNRGBD config
    with open(mini_config_path, 'r') as f:
        mini_config = json.load(f)

    # Filter classes by minimum sample threshold
    valid_classes = {}
    excluded_classes = {}
    for class_name, sample_ids in mini_config.items():
        if len(sample_ids) >= min_samples:
            valid_classes[class_name] = sample_ids
        else:
            excluded_classes[class_name] = len(sample_ids)

    print(f"Valid classes ({len(valid_classes)}): {list(valid_classes.keys())}")
    print(f"Excluded classes (< {min_samples}): {excluded_classes}")

    # Random sampling for large classes
    sampled_classes = {}
    for class_name, sample_ids in valid_classes.items():
        if len(sample_ids) > sample_limit:
            sampled_ids = random.sample(sample_ids, sample_limit)
            print(f"  {class_name}: {len(sample_ids)} -> {len(sampled_ids)} (sampled)")
        else:
            sampled_ids = sample_ids
            print(f"  {class_name}: {len(sample_ids)} (kept all)")
        sampled_classes[class_name] = sampled_ids

    # Build sample_id to valid class_names mapping
    target_sample_ids = set()
    sample_to_classes = defaultdict(set)
    for class_name, sample_ids in sampled_classes.items():
        for sid in sample_ids:
            target_sample_ids.add(sid)
            sample_to_classes[sid].add(class_name)

    print(f"\nTotal target samples: {len(target_sample_ids)}")

    # Statistics
    stats = {
        'valid_classes': list(valid_classes.keys()),
        'excluded_classes': list(excluded_classes.keys()),
        'total_target_samples': len(target_sample_ids),
        'per_class_count': {c: len(v) for c, v in sampled_classes.items()},
    }

    # Label file directory
    label_dir = osp.join(root_path, 'sunrgbd_trainval', 'label')

    # Process train and val splits
    for split in ['train', 'val']:
        pkl_path = osp.join(root_path, f'sunrgbd_infos_{split}.pkl')
        if not osp.exists(pkl_path):
            print(f"\nWarning: {pkl_path} not found, skipping")
            continue

        print(f"\nProcessing {split} split...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Separate target samples and potential negative samples
        target_infos = []
        negative_candidates = []

        for info in data['data_list']:
            lidar_path = info['lidar_points']['lidar_path']
            sample_idx = osp.splitext(osp.basename(lidar_path))[0]

            if sample_idx in target_sample_ids:
                # This is a target sample - parse label file for instances
                valid_class_names = sample_to_classes[sample_idx]
                label_file = osp.join(label_dir, f'{sample_idx}.txt')

                instances = []
                if osp.exists(label_file):
                    with open(label_file, 'r') as f:
                        label_lines = f.readlines()

                    for line in label_lines:
                        parsed = parse_sunrgbd_label_line(line)
                        if parsed is None:
                            continue
                        if parsed['classname'] in valid_class_names:
                            label = MINI_CLASS_TO_LABEL[parsed['classname']]
                            instances.append(create_instance_dict(parsed, label, parsed['classname']))

                # Only keep samples with at least one valid instance
                if len(instances) > 0:
                    info_copy = info.copy()
                    info_copy['instances'] = instances
                    target_infos.append(info_copy)
            else:
                # Potential negative sample
                negative_candidates.append(info)

        # Add negative samples (~10% of target samples)
        num_negative = int(len(target_infos) * negative_ratio)
        if len(negative_candidates) > 0 and num_negative > 0:
            # Filter out samples that have any target classes
            truly_negative = []
            for info in negative_candidates:
                lidar_path = info['lidar_points']['lidar_path']
                sample_idx = osp.splitext(osp.basename(lidar_path))[0]
                label_file = osp.join(label_dir, f'{sample_idx}.txt')

                has_target_class = False
                if osp.exists(label_file):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parsed = parse_sunrgbd_label_line(line)
                            if parsed and parsed['classname'] in MINI_CLASS_TO_LABEL:
                                has_target_class = True
                                break

                if not has_target_class:
                    info_copy = info.copy()
                    info_copy['instances'] = []  # Empty instances for negative sample
                    truly_negative.append(info_copy)

            # Random select negative samples
            if len(truly_negative) > num_negative:
                selected_negative = random.sample(truly_negative, num_negative)
            else:
                selected_negative = truly_negative
                print(f"  Warning: Only {len(truly_negative)} negative samples available, requested {num_negative}")
        else:
            selected_negative = []

        filtered_data_list = target_infos + selected_negative
        random.shuffle(filtered_data_list)

        print(f"  Target samples: {len(target_infos)}")
        print(f"  Negative samples: {len(selected_negative)}")
        print(f"  Total: {len(filtered_data_list)}")

        # Count instances per class
        class_counts = defaultdict(int)
        for info in filtered_data_list:
            for inst in info.get('instances', []):
                class_counts[inst['class_name']] += 1
        print(f"  Instances per class: {dict(class_counts)}")

        # Create output data
        output_data = {
            'metainfo': {
                'categories': MINI_CLASS_TO_LABEL,
                'dataset': 'mini_sunrgbd',
                'info_version': '1.0',
            },
            'data_list': filtered_data_list,
        }

        # Save filtered pkl
        output_path = osp.join(output_dir, f'mini_sunrgbd_infos_{split}.pkl')
        mmengine.mkdir_or_exist(output_dir)
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"  Saved to: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Convert SUNRGBD to MiniSUNRGBD')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data2/sunrgbd',
        help='Root path of original SUNRGBD data',
    )
    parser.add_argument(
        '--mini-config',
        type=str,
        default='~/ros2_ws/MiniSUNRGBD.json',
        help='Path to MiniSUNRGBD.json config',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data2/mini_sunrgbd',
        help='Output directory',
    )
    parser.add_argument(
        '--sample-limit',
        type=int,
        default=50,
        help='Maximum samples per class',
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=18,
        help='Minimum samples threshold to keep a class',
    )
    parser.add_argument(
        '--negative-ratio',
        type=float,
        default=0.1,
        help='Ratio of negative samples to add',
    )
    args = parser.parse_args()

    # Expand user path
    mini_config = osp.expanduser(args.mini_config)

    print("=" * 60)
    print("MiniSUNRGBD Dataset Converter")
    print("=" * 60)
    print(f"Root path: {args.root_path}")
    print(f"Mini config: {mini_config}")
    print(f"Output dir: {args.out_dir}")
    print(f"Sample limit: {args.sample_limit}")
    print(f"Min samples: {args.min_samples}")
    print(f"Negative ratio: {args.negative_ratio}")
    print("=" * 60)

    stats = filter_mini_sunrgbd_infos(
        root_path=args.root_path,
        mini_config_path=mini_config,
        output_dir=args.out_dir,
        sample_limit=args.sample_limit,
        min_samples=args.min_samples,
        negative_ratio=args.negative_ratio,
    )

    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print(f"Valid classes: {stats['valid_classes']}")
    print(f"Total target samples: {stats['total_target_samples']}")
    print(f"Per-class counts: {stats['per_class_count']}")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""将 SUNRGBD 数据集转换为 MiniSUNRGBD 数据集，用于桌面物体检测。

MiniSUNRGBD 是 SUNRGBD 的子集，包含 8 个桌面物体类别：
keyboard, laptop, book, cup, mug, pen, notebook, phone

此转换器：
1. 加载 MiniSUNRGBD.json 获取目标类别和样本 ID
2. 将类别重新索引为 0-7 用于 MiniSUNRGBD
3. 可选地从 "_" 配置添加负样本（基于正样本比例）
4. 生成带有新标签索引的 pkl 文件
5. 从实际 bbox 标注计算 mean_sizes
"""

import argparse
import json
import os
import os.path as osp
from pathlib import Path
import pickle
import random
import shutil
from collections import defaultdict

import mmengine
import numpy as np

# MiniSUNRGBD 类别顺序（必须与数据集 METAINFO 匹配）
MINI_SUNRGBD_CLASSES = set(['keyboard', 'laptop', 'book', 'cup', 'mug',
                         'pen', 'notebook', 'phone'])
MINI_CLASS_TO_LABEL = {c: i for i, c in enumerate(MINI_SUNRGBD_CLASSES)}

# 类别名到 mean_size 索引的映射（用于输出）
CLASS_TO_MEAN_SIZE_IDX = {c: i for i, c in enumerate(sorted(MINI_SUNRGBD_CLASSES))}


def parse_sunrgbd_label_line(line: str) -> dict:
    """解析 SUNRGBD label 文件的单行数据。

    标签格式：
    classname xmin ymin xmax ymax centroid_x centroid_y centroid_z \
        width length height orientation_x orientation_y

    返回:
        解析后的字段字典，解析失败返回 None
    """
    parts = line.strip().split()
    if len(parts) < 13:
        return None

    try:
        data = [float(x) for x in parts[1:]]
    except ValueError:
        return None

    # heading_angle = arctan2(orientation_y, orientation_x) = arctan2(data[11], data[10])
    # 但某些行格式可能不同，使用安全访问方式
    if len(data) >= 12:
        heading_angle = np.arctan2(data[11], data[10])
    elif len(data) >= 11:
        heading_angle = data[10]
    else:
        heading_angle = 0.0

    return {
        'classname': parts[0],
        'bbox': np.array([data[0], data[1], data[0] + data[3], data[2] + data[4]]),  # x1, y1, x2, y2
        'center': np.array([data[4], data[5], data[6]]),  # cx, cy, cz
        'size': np.array([data[9], data[8], data[10]]) * 2,  # length, width, height (x_size, y_size, z_size)
        'heading_angle': heading_angle,
    }


def clean_label_file(label_path: str) -> int:
    """清洗label txt文件，只保留有效类别的行。

    Args:
        label_path: label文件路径

    Returns:
        被剔除的行数
    """
    if not osp.exists(label_path):
        return 0

    with open(label_path, 'r') as f:
        lines = f.readlines()

    valid_lines = []
    removed_count = 0
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0] in MINI_SUNRGBD_CLASSES:
            valid_lines.append(line)
        else:
            removed_count += 1

    # 重写文件
    with open(label_path, 'w') as f:
        f.writelines(valid_lines)

    return removed_count


def copy_mini_sunrgbd_files(src_root: Path, dst_root: Path, sample_ids: set) -> dict:
    """复制 MiniSUNRGBD 子集文件到目标目录。

    Args:
        src_root: 源数据集根目录 (sunrgbd)
        dst_root: 目标数据集根目录 (mini_sunrgbd)
        sample_ids: 要复制的样本ID集合

    Returns:
        统计信息字典
    """
    # 创建目录结构
    (dst_root / 'points').mkdir(parents=True, exist_ok=True)
    (dst_root / 'sunrgbd_trainval/image').mkdir(parents=True, exist_ok=True)
    (dst_root / 'sunrgbd_trainval/calib').mkdir(parents=True, exist_ok=True)
    (dst_root / 'sunrgbd_trainval/label').mkdir(parents=True, exist_ok=True)

    # 定义要复制的文件
    files_to_copy = [
        ('points/{}.bin', 'points/{}.bin'),
        ('sunrgbd_trainval/image/{}.jpg', 'sunrgbd_trainval/image/{}.jpg'),
        ('sunrgbd_trainval/calib/{}.txt', 'sunrgbd_trainval/calib/{}.txt'),
        ('sunrgbd_trainval/label/{}.txt', 'sunrgbd_trainval/label/{}.txt'),
    ]

    stats = {
        'copied': 0,
        'missing': 0,
        'exist': 0,
        'labels_cleaned': 0,
        'labels_removed_lines': 0,
    }

    for idx in sample_ids:
        for src_path, dst_path in files_to_copy:
            src = src_root / src_path.format(idx)
            dst = dst_root / dst_path.format(idx)
            if dst.exists():
                stats['exist'] += 1
                continue
            if src.exists():
                shutil.copy2(src, dst)
                stats['copied'] += 1
            else:
                stats['missing'] += 1

        # 清洗 label 文件
        label_file = dst_root / 'sunrgbd_trainval/label/{}.txt'.format(idx)
        if label_file.exists():
            removed = clean_label_file(str(label_file))
            if removed > 0:
                stats['labels_cleaned'] += 1
                stats['labels_removed_lines'] += removed

    return stats


def create_instance_dict(parsed: dict, label: int, classname: str) -> dict:
    """创建实例字典，3D bbox 格式为 [cx, cy, cz, l, w, h, angle]。"""
    size = parsed['size'].copy()
    box3d = np.concatenate([
        parsed['center'],      # cx, cy, cz
        size,                  # l, w, h
        np.array([parsed['heading_angle']])  # angle
    ])
    return {
        'bbox': parsed['bbox'],          # 2D bbox [x1, y1, x2, y2]
        'bbox_label': label,             # 2D 标签（与 3D 相同）
        'bbox_3d': box3d.astype(np.float32),  # 3D bbox [cx, cy, cz, l, w, h, angle]
        'bbox_label_3d': label,          # 3D 标签（MiniSUNRGBD 索引 0-7）
        'class_name': classname,         # 原始类别名称
    }


def compute_mean_sizes(class_sizes: dict) -> list:
    """从收集的 bbox 尺寸计算每个类别的平均尺寸。

    Args:
        class_sizes: 字典，键为 class_name，值为 (l, w, h) 元组列表

    Returns:
        按类别排序的的平均尺寸列表 [l, w, h]
    """
    mean_sizes = []
    for class_name in sorted(MINI_SUNRGBD_CLASSES):
        sizes = class_sizes.get(class_name, [])
        if len(sizes) > 0:
            sizes_arr = np.array(sizes)
            mean_l = float(sizes_arr[:, 0].mean())
            mean_w = float(sizes_arr[:, 1].mean())
            mean_h = float(sizes_arr[:, 2].mean())
            mean_sizes.append([mean_l, mean_w, mean_h])
            print(f"  {class_name}: {len(sizes)} samples, mean_size=[{mean_l:.3f}, {mean_w:.3f}, {mean_h:.3f}]")
        else:
            mean_sizes.append([0.0, 0.0, 0.0])
            print(f"  {class_name}: no samples (using default [0,0,0])")
    return mean_sizes


def filter_mini_sunrgbd_infos(
    root_path: str,
    mini_config_path: str,
    output_dir: str,
    sample_limit: int = 50,
    min_samples: int = 18,
    negative_ratio: float = 0.15,
    compute_stats: bool = True,
) -> dict:
    """从完整 SUNRGBD 数据集中筛选 MiniSUNRGBD 子集。

    Args:
        root_path: 原始 SUNRGBD 数据的根目录路径。
        mini_config_path: MiniSUNRGBD.json 配置文件路径。
        output_dir: 过滤后 pkl 文件的输出目录。
        sample_limit: 每个类别的最大样本数（默认 50）。
        min_samples: 保留类别的最小样本阈值（默认 18）。
        negative_ratio: 从 "_" 配置添加负样本的比例。
            0 表示不添加负样本（默认 0.15 = 正样本数量的 15%）。

    Returns:
        包含过滤后数据集信息的统计字典。
    """
    # 加载 MiniSUNRGBD 配置
    with open(mini_config_path, 'r') as f:
        mini_config = json.load(f)

    # 获取下划线样本（来自 SUNRGBD 的非目标样本）
    underscore_samples = set(mini_config.get('_', []))
    print(f"下划线样本数量: {len(underscore_samples)}")

    # 按最小样本阈值过滤类别（排除 "_" 和其他非类别条目）
    valid_classes = {}
    excluded_classes = {}
    for class_name, sample_ids in mini_config.items():
        # 跳过 "_" 和其他非类别条目
        if class_name == '_' or class_name not in MINI_CLASS_TO_LABEL:
            continue
        if len(sample_ids) >= min_samples:
            valid_classes[class_name] = sample_ids
        else:
            excluded_classes[class_name] = len(sample_ids)

    print(f"有效类别 ({len(valid_classes)}): {list(valid_classes.keys())}")
    print(f"排除的类别 (< {min_samples}): {excluded_classes}")

    # 对大类进行随机采样
    sampled_classes = {}
    for class_name, sample_ids in valid_classes.items():
        if len(sample_ids) > sample_limit:
            sampled_ids = random.sample(sample_ids, sample_limit)
            print(f"  {class_name}: {len(sample_ids)} -> {len(sampled_ids)} (sampled)")
        else:
            sampled_ids = sample_ids
            print(f"  {class_name}: {len(sample_ids)} (kept all)")
        sampled_classes[class_name] = sampled_ids

    # 构建 sample_id 到有效类别名的映射
    target_sample_ids = set()
    sample_to_classes = defaultdict(set)
    for class_name, sample_ids in sampled_classes.items():
        for sid in sample_ids:
            target_sample_ids.add(sid)
            sample_to_classes[sid].add(class_name)

    print(f"\n目标样本总数: {len(target_sample_ids)}")

    # 统计信息
    stats = {
        'valid_classes': list(valid_classes.keys()),
        'excluded_classes': list(excluded_classes.keys()),
        'total_target_samples': len(target_sample_ids),
        'per_class_count': {c: len(v) for c, v in sampled_classes.items()},
    }

    # 收集每个类别的 bbox 尺寸用于计算 mean_sizes
    class_bbox_sizes = defaultdict(list)

    # 标签文件目录
    label_dir = osp.join(root_path, 'sunrgbd_trainval', 'label')

    # 处理 train 和 val 划分
    for split in ['train', 'val']:
        pkl_path = osp.join(root_path, f'sunrgbd_infos_{split}.pkl')
        if not osp.exists(pkl_path):
            print(f"\n警告: {pkl_path} 未找到，跳过")
            continue

        print(f"\n正在处理 {split} 划分...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 分离目标样本和潜在负样本
        target_infos = []
        negative_candidates = []

        for info in data['data_list']:
            lidar_path = info['lidar_points']['lidar_path']
            sample_idx = osp.splitext(osp.basename(lidar_path))[0]

            if sample_idx in target_sample_ids:
                # 这是目标样本 - 解析标签文件获取实例
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
                            # 收集 bbox 尺寸用于计算 mean_sizes
                            size = parsed['size'].copy()
                            class_bbox_sizes[parsed['classname']].append(size)

                # 只保留至少有一个有效实例的样本
                if len(instances) > 0:
                    info_copy = info.copy()
                    info_copy['instances'] = instances
                    target_infos.append(info_copy)
            else:
                # 潜在负样本
                negative_candidates.append(info)

        # 从下划线配置添加负样本
        selected_negative = []
        if negative_ratio > 0 and underscore_samples:
            num_negative = int(len(target_infos) * negative_ratio)

            for info in data['data_list']:
                lidar_path = info['lidar_points']['lidar_path']
                sample_idx = osp.splitext(osp.basename(lidar_path))[0]
                if sample_idx in underscore_samples:
                    info_copy = info.copy()
                    info_copy['instances'] = []  # 负样本的空实例
                    selected_negative.append(info_copy)

            print(f"  找到的下划线样本: {len(selected_negative)}")
            print(f"  请求的负样本数: {num_negative}")

            # 如果太多则随机选择
            if len(selected_negative) > num_negative:
                selected_negative = random.sample(selected_negative, num_negative)
        elif negative_ratio == 0:
            print("  negative_ratio=0，不添加负样本")

        filtered_data_list = target_infos + selected_negative
        random.shuffle(filtered_data_list)

        print(f"  目标样本: {len(target_infos)}")
        print(f"  负样本: {len(selected_negative)}")
        print(f"  总计: {len(filtered_data_list)}")

        # 统计每个类别的实例数
        class_counts = defaultdict(int)
        for info in filtered_data_list:
            for inst in info.get('instances', []):
                class_counts[inst['class_name']] += 1
        print(f"  每个类别的实例数: {dict(class_counts)}")

        # 创建输出数据
        output_data = {
            'metainfo': {
                'categories': MINI_CLASS_TO_LABEL,
                'dataset': 'mini_sunrgbd',
                'info_version': '1.0',
            },
            'data_list': filtered_data_list,
        }

        # 保存过滤后的 pkl
        output_path = osp.join(output_dir, f'mini_sunrgbd_infos_{split}.pkl')
        mmengine.mkdir_or_exist(output_dir)
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"  保存至: {output_path}")

    # 从收集的 bbox 尺寸计算 mean_sizes
    if compute_stats:
        print("\n" + "=" * 60)
        print("从标注计算 mean_sizes:")
        mean_sizes = compute_mean_sizes(class_bbox_sizes)
        stats['mean_sizes'] = mean_sizes

        # 保存 mean_sizes 到 JSON 文件以便参考
        mean_sizes_path = osp.join(output_dir, 'mean_sizes.json')
        with open(mean_sizes_path, 'w') as f:
            json.dump({
                'mean_sizes': mean_sizes,
                'class_order': sorted(MINI_SUNRGBD_CLASSES),
                'per_class_sample_count': {c: len(v) for c, v in class_bbox_sizes.items()}
            }, f, indent=2)
        print(f"Mean sizes 已保存至: {mean_sizes_path}")
        print("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(description='将 SUNRGBD 转换为 MiniSUNRGBD')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data2/sunrgbd',
        help='原始 SUNRGBD 数据的根目录',
    )
    parser.add_argument(
        '--mini-config',
        type=str,
        default='./data2/mini_sunrgbd/MiniSUNRGBD.json',
        help='MiniSUNRGBD.json 配置文件路径',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data2/mini_sunrgbd',
        help='输出目录',
    )
    parser.add_argument(
        '--sample-limit',
        type=int,
        default=50,
        help='每个类别的最大样本数',
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=18,
        help='保留类别的最小样本阈值',
    )
    parser.add_argument(
        '--negative-ratio',
        type=float,
        default=0.15,
        help='从 "_" 配置添加负样本的比例（默认 0.15 = 正样本的 15%%，0 = 不添加负样本）',
    )
    args = parser.parse_args()

    # 扩展用户路径
    mini_config = osp.expanduser(args.mini_config)

    print("=" * 60)
    print("MiniSUNRGBD 数据集转换器")
    print("=" * 60)
    print(f"根目录: {args.root_path}")
    print(f"Mini 配置: {mini_config}")
    print(f"输出目录: {args.out_dir}")
    print(f"样本上限: {args.sample_limit}")
    print(f"最小样本数: {args.min_samples}")
    print(f"负样本比例: {args.negative_ratio}")
    print("=" * 60)
    
    assert Path(args.root_path).exists(), "SUN RGBD 数据集为找到"

    # 加载配置获取所有样本ID（包括underscore）
    with open(mini_config, 'r') as f:
        mini_config_data = json.load(f)

    # 收集所有要复制的样本ID
    all_sample_ids = set()
    for class_name, sample_ids in mini_config_data.items():
        if class_name == '_' or class_name not in MINI_CLASS_TO_LABEL:
            continue
        all_sample_ids.update(sample_ids)
    # 也包含underscore样本
    all_sample_ids.update(mini_config_data.get('_', []))

    print(f"\n复制文件阶段:")
    print(f"  总样本数: {len(all_sample_ids)}")
    copy_stats = copy_mini_sunrgbd_files(
        src_root=Path(args.root_path),
        dst_root=Path(args.out_dir),
        sample_ids=all_sample_ids,
    )
    print(f"  复制成功: {copy_stats['copied']}")
    print(f"  文件缺失: {copy_stats['missing']}")
    print(f"  文件已存在: {copy_stats['exist']}")
    print(f"  清洗label文件: {copy_stats['labels_cleaned']}")
    print(f"  剔除无效行: {copy_stats['labels_removed_lines']}")

    print(f"\n生成PKL阶段:")
    stats = filter_mini_sunrgbd_infos(
        root_path=args.root_path,
        mini_config_path=mini_config,
        output_dir=args.out_dir,
        sample_limit=args.sample_limit,
        min_samples=args.min_samples,
        negative_ratio=args.negative_ratio,
    )

    print("\n" + "=" * 60)
    print("转换完成！")
    print(f"有效类别: {stats['valid_classes']}")
    print(f"目标样本总数: {stats['total_target_samples']}")
    print(f"每类别计数: {stats['per_class_count']}")
    print("=" * 60)


if __name__ == '__main__':
    main()

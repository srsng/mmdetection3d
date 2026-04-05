"""将TO-Scene mean_sizes npz文件转换为py列表

使用方法:
    python extract_mean_sizes.py
"""

import numpy as np

# 70类完整
classes = [
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
]

# 52类小物体名称 (对应70类中索引18-69)
classes_52 = classes.copy()[18:]


def convert_mean_sizes(input_path, output_path, is_52class=False):
    """转换mean_sizes文件

    Args:
        input_path: 输入npz路径
        output_path: 输出py路径
        is_52class: 是否只提取52类小物体
    """
    # 加载npz
    data = np.load(input_path)
    mean_sizes = data["arr_0"]

    if is_52class:
        # 提取52类 (索引18-69)
        mean_sizes = mean_sizes[18:]
        names = classes_52
    else:
        names = classes

    # 构建
    lines = []
    for i, name in enumerate(names):
        ms = mean_sizes[i]
        lines.append(
            f"[{float(ms[0]):06f}, {float(ms[1]):06f}, {float(ms[2]):06f}],  # {name} ({i})"
        )

    # 保存
    with open(output_path, "w") as f:
        data = "\n    ".join(lines)
        f.write(f"mean_sizes = [\n    {data}\n]")

    print(f"Saved {len(lines)} classes to {output_path}")


def main():
    base_dir = "./data2/TO-SCENE-down/TO-scannet"
    output_dir = "./data2/TO-SCENE-down/TO-scannet"

    # 转换52类
    convert_mean_sizes(
        f"{base_dir}/means.npz",
        f"{output_dir}/mean_sizes.py",
        is_52class=False,
    )


if __name__ == "__main__":
    main()

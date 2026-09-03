from pathlib import Path
from io import BytesIO

from datasets import load_dataset
from PIL import Image


PARQUET_DIR = Path(r"S:\简历\字节\gundam_parquet")
OUTPUT_DIR = Path(r"S:\简历\字节\training_images")
IMAGE_COLUMN = "image"


def open_image(value):
    if isinstance(value, Image.Image):
        return value

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")

        if image_bytes:
            return Image.open(BytesIO(image_bytes))

        if image_path:
            return Image.open(image_path)

    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value))

    if isinstance(value, str):
        return Image.open(value)

    raise TypeError(f"无法识别图片数据类型：{type(value)}")


parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(
        f"在 {PARQUET_DIR} 中没有找到任何 .parquet 文件"
    )

print("找到以下 Parquet 文件：")

for file in parquet_files:
    print(file.name)

dataset = load_dataset(
    "parquet",
    data_files=[str(file) for file in parquet_files],
    split="train",
)

print(f"\n数据集共有 {len(dataset)} 行")
print(f"数据列：{dataset.column_names}")

if IMAGE_COLUMN not in dataset.column_names:
    raise KeyError(
        f"没有找到名为 '{IMAGE_COLUMN}' 的图片列。\n"
        f"当前数据列为：{dataset.column_names}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

success_count = 0
failed_count = 0

for index, row in enumerate(dataset):
    try:
        image = open_image(row[IMAGE_COLUMN])
        image = image.convert("RGB")

        output_path = OUTPUT_DIR / f"gundam_{index:05d}.jpg"
        image.save(output_path, format="JPEG", quality=95)

        success_count += 1

        if success_count % 100 == 0:
            print(f"已经提取 {success_count} 张图片")

    except Exception as error:
        failed_count += 1
        print(f"第 {index} 行提取失败：{error}")

print("\n提取完成")
print(f"成功：{success_count} 张")
print(f"失败：{failed_count} 张")
print(f"图片位置：{OUTPUT_DIR}")
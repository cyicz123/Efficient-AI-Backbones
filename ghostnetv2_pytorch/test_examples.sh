#!/bin/bash

# GhostNetV2 模型测试示例脚本
# 这个脚本展示了如何使用 test.py 来测试模型性能

# 设置基本路径
DATA_PATH="/path/to/your/dataset"  # 修改为你的数据集路径
CHECKPOINT_PATH="/path/to/your/checkpoint.pth"  # 修改为你的权重文件路径
OUTPUT_PATH="./test_results"  # 测试结果输出路径

# 基本测试命令
echo "开始基本测试..."
python test.py \
    $DATA_PATH \
    --model ghostnetv2 \
    --checkpoint $CHECKPOINT_PATH \
    --batch-size 128 \
    --num-classes 1000 \
    --output $OUTPUT_PATH

# 高精度测试（使用TTA）
echo "开始高精度测试（使用TTA）..."
python test.py \
    $DATA_PATH \
    --model ghostnetv2 \
    --checkpoint $CHECKPOINT_PATH \
    --batch-size 64 \
    --num-classes 1000 \
    --tta 3 \
    --output ${OUTPUT_PATH}_tta \
    --save-results

# 自定义图像尺寸测试
echo "开始自定义图像尺寸测试..."
python test.py \
    $DATA_PATH \
    --model ghostnetv2 \
    --checkpoint $CHECKPOINT_PATH \
    --batch-size 96 \
    --img-size 256 \
    --crop-pct 0.9 \
    --num-classes 1000 \
    --output ${OUTPUT_PATH}_256

# 不同宽度模型测试
echo "开始不同宽度模型测试..."
for width in 0.5 0.75 1.0 1.25; do
    echo "测试宽度: $width"
    python test.py \
        $DATA_PATH \
        --model ghostnetv2 \
        --checkpoint $CHECKPOINT_PATH \
        --batch-size 128 \
        --width $width \
        --num-classes 1000 \
        --output ${OUTPUT_PATH}_width_$width
done

echo "所有测试完成！"

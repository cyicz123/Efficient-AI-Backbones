# GhostNetV2 模型测试脚本使用说明

本项目提供了两个测试脚本来评估GhostNetV2模型在数据集上的性能：

## 文件说明

1. **test.py** - 完整功能的测试脚本（基于timm库）
2. **simple_test.py** - 简化版测试脚本（使用PyTorch和torchvision）
3. **test_examples.sh** - 测试示例脚本

## 环境要求

### 对于 test.py（推荐）
```bash
pip install torch torchvision timm PyYAML
```

### 对于 simple_test.py
```bash
pip install torch torchvision PyYAML
```

## 数据集格式

测试脚本支持ImageNet格式的数据集，目录结构应为：
```
dataset/
├── val/          # 或 validation/ 或 test/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── class2/
│   │   ├── img3.jpg
│   │   └── img4.jpg
│   └── ...
```

## 使用方法

### 1. 基本测试（推荐使用 simple_test.py）

```bash
python simple_test.py /path/to/dataset --checkpoint /path/to/model.pth
```

### 2. 完整功能测试（使用 test.py）

```bash
python test.py /path/to/dataset --checkpoint /path/to/model.pth --model ghostnetv2
```

### 3. 自定义参数测试

```bash
# 指定批次大小和图像尺寸
python simple_test.py /path/to/dataset \
    --checkpoint /path/to/model.pth \
    --batch-size 128 \
    --img-size 256

# 测试不同宽度的模型
python simple_test.py /path/to/dataset \
    --checkpoint /path/to/model.pth \
    --width 0.75 \
    --num-classes 1000
```

### 4. 高精度测试（仅test.py支持）

```bash
# 使用测试时增强（TTA）
python test.py /path/to/dataset \
    --checkpoint /path/to/model.pth \
    --model ghostnetv2 \
    --tta 3 \
    --save-results \
    --output ./test_results
```

## 参数说明

### 通用参数
- `data`: 数据集路径（必需）
- `--checkpoint`: 模型权重文件路径（必需）
- `--batch-size`: 批次大小（默认：256）
- `--img-size`: 输入图像尺寸（默认：224）
- `--num-classes`: 分类数量（默认：1000）
- `--width`: 模型宽度倍数（默认：1.0）
- `--workers`: 数据加载进程数（默认：4）
- `--device`: 设备选择，cuda或cpu（默认：cuda）

### test.py 专有参数
- `--model`: 模型名称（默认：ghostnetv2）
- `--tta`: 测试时增强次数（默认：0）
- `--save-results`: 保存详细测试结果
- `--output`: 结果输出路径
- `--amp`: 使用混合精度推理

## 输出结果

测试完成后会显示：
- 测试时间
- 样本数量
- 平均推理时间
- Top-1 准确率
- Top-5 准确率

如果使用 `--output` 参数，还会保存：
- `test_summary.yaml`: 测试结果摘要
- `detailed_results.json`: 详细结果（需要 `--save-results`）

## 示例输出

```
2024-XX-XX XX:XX:XX - INFO - 使用设备: cuda
2024-XX-XX XX:XX:XX - INFO - 正在加载模型...
2024-XX-XX XX:XX:XX - INFO - 加载权重文件: /path/to/model.pth
2024-XX-XX XX:XX:XX - INFO - 权重加载完成
2024-XX-XX XX:XX:XX - INFO - 模型参数数量: 6,209,048
2024-XX-XX XX:XX:XX - INFO - 正在准备数据...
2024-XX-XX XX:XX:XX - INFO - 使用数据路径: /path/to/dataset/val
2024-XX-XX XX:XX:XX - INFO - 数据集大小: 50000
2024-XX-XX XX:XX:XX - INFO - 类别数量: 1000
2024-XX-XX XX:XX:XX - INFO - 开始测试...
============================================================
测试完成!
数据集: /path/to/dataset
权重文件: /path/to/model.pth
样本数量: 50000
批次大小: 256
图像尺寸: 224
模型宽度: 1.0
测试时间: 123.45秒
平均推理时间: 2.47ms/样本
Top-1 准确率: 75.2340%
Top-5 准确率: 92.1250%
============================================================
```

## 常见问题

### 1. CUDA内存不足
减小批次大小：
```bash
python simple_test.py /path/to/dataset --checkpoint /path/to/model.pth --batch-size 64
```

### 2. 权重文件加载失败
确保权重文件格式正确，脚本会自动处理以下格式：
- 直接的state_dict
- 包含'state_dict'键的字典
- 包含'model'键的字典

### 3. 数据集路径错误
确保数据集路径包含以下子目录之一：
- `val/`
- `validation/`
- `test/`

或者直接使用包含分类子目录的路径。

## 批量测试

使用提供的 `test_examples.sh` 脚本进行批量测试：

```bash
# 修改脚本中的路径
vim test_examples.sh

# 运行批量测试
chmod +x test_examples.sh
./test_examples.sh
```

## 性能优化建议

1. **使用更大的批次大小**：在GPU内存允许的情况下
2. **使用更多的数据加载进程**：根据CPU核数调整 `--workers`
3. **使用混合精度**：添加 `--amp` 参数（仅test.py）
4. **固定内存**：添加 `--pin-mem` 参数（仅test.py）

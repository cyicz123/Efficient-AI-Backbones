#!/usr/bin/env python
"""
简化版GhostNetV2模型测试脚本
用于快速测试模型在数据集上的性能
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入模型
from model.ghostnetv2_torch import ghostnetv2

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """计算指定topk的准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_model(checkpoint_path, num_classes=1000, width=1.0, device='cuda'):
    """加载模型和权重"""
    # 创建模型
    model = ghostnetv2(num_classes=num_classes, width=width, dropout=0.0, args=None)
    
    # 加载权重
    if os.path.isfile(checkpoint_path):
        logger.info(f'加载权重文件: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同格式的权重文件
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 处理权重键名
        model_state_dict = model.state_dict()
        new_state_dict = {}
        
        for k, v in state_dict.items():
            # 移除 'module.' 前缀（如果存在）
            name = k[7:] if k.startswith('module.') else k
            if name in model_state_dict:
                new_state_dict[name] = v
            else:
                logger.warning(f"权重文件中的键 '{name}' 在模型中不存在")
        
        model.load_state_dict(new_state_dict, strict=False)
        logger.info('权重加载完成')
    else:
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
    
    model = model.to(device)
    return model

def create_data_loader(data_path, batch_size=256, img_size=224, num_workers=4):
    """创建数据加载器"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(int(img_size / 0.875)),  # 调整大小
        transforms.CenterCrop(img_size),           # 中心裁剪
        transforms.ToTensor(),                     # 转为Tensor
        transforms.Normalize(                      # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 寻找验证集路径
    val_path = None
    for subdir in ['val', 'validation', 'test']:
        potential_path = os.path.join(data_path, subdir)
        if os.path.isdir(potential_path):
            val_path = potential_path
            break
    
    if val_path is None:
        # 如果没有子目录，直接使用数据路径
        val_path = data_path
    
    logger.info(f'使用数据路径: {val_path}')
    
    # 创建数据集
    dataset = datasets.ImageFolder(val_path, transform=transform)
    logger.info(f'数据集大小: {len(dataset)}')
    logger.info(f'类别数量: {len(dataset.classes)}')
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return data_loader, len(dataset)

def test_model(model, data_loader, device='cuda', log_interval=50):
    """测试模型"""
    model.eval()
    
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # 数据移到GPU
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # 前向传播
            output = model(data)
            
            # 计算准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            # 更新统计
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
            
            # 测量时间
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 打印进度
            if batch_idx % log_interval == 0:
                logger.info(
                    f'Test: [{batch_idx}/{len(data_loader)}] '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                    f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
                )
    
    return top1.avg, top5.avg

def main():
    parser = argparse.ArgumentParser(description='GhostNetV2 简化测试脚本')
    parser.add_argument('data', help='数据集路径')
    parser.add_argument('--checkpoint', required=True, help='模型权重文件路径')
    parser.add_argument('--batch-size', type=int, default=256, help='批次大小')
    parser.add_argument('--img-size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--num-classes', type=int, default=1000, help='类别数量')
    parser.add_argument('--width', type=float, default=1.0, help='模型宽度倍数')
    parser.add_argument('--workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--device', default='cuda', help='设备')
    parser.add_argument('--log-interval', type=int, default=50, help='日志间隔')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA不可用，使用CPU')
        args.device = 'cpu'
    
    logger.info(f'使用设备: {args.device}')
    
    # 加载模型
    logger.info('正在加载模型...')
    model = load_model(args.checkpoint, args.num_classes, args.width, args.device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'模型参数数量: {total_params:,}')
    
    # 创建数据加载器
    logger.info('正在准备数据...')
    data_loader, num_samples = create_data_loader(
        args.data, args.batch_size, args.img_size, args.workers
    )
    
    # 开始测试
    logger.info('开始测试...')
    start_time = time.time()
    
    top1_acc, top5_acc = test_model(model, data_loader, args.device, args.log_interval)
    
    end_time = time.time()
    test_time = end_time - start_time
    
    # 打印结果
    logger.info('=' * 60)
    logger.info('测试完成!')
    logger.info(f'数据集: {args.data}')
    logger.info(f'权重文件: {args.checkpoint}')
    logger.info(f'样本数量: {num_samples}')
    logger.info(f'批次大小: {args.batch_size}')
    logger.info(f'图像尺寸: {args.img_size}')
    logger.info(f'模型宽度: {args.width}')
    logger.info(f'测试时间: {test_time:.2f}秒')
    logger.info(f'平均推理时间: {test_time/num_samples*1000:.2f}ms/样本')
    logger.info(f'Top-1 准确率: {top1_acc:.4f}%')
    logger.info(f'Top-5 准确率: {top5_acc:.4f}%')
    logger.info('=' * 60)

if __name__ == '__main__':
    main()

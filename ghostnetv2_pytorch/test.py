#!/usr/bin/env python
"""
测试脚本用于GhostNetV2模型在数据集上的性能评估
基于训练脚本修改而来，专门用于模型测试
"""

import os
import argparse
import time
import yaml
import warnings
import logging
import torch
import torch.nn as nn
from collections import OrderedDict

warnings.filterwarnings('ignore')

from timm.data import Dataset, create_loader, resolve_data_config
from timm.models import create_model
from timm.utils import *

from model import ghostnetv2_torch

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args_parser():
    parser = argparse.ArgumentParser(description='GhostNetV2 模型测试脚本')
    
    # 数据集参数
    parser.add_argument('data', metavar='DIR', help='数据集路径')
    parser.add_argument('--model', default='ghostnetv2', type=str, metavar='MODEL',
                       help='模型名称 (默认: "ghostnetv2")')
    parser.add_argument('--checkpoint', required=True, type=str, metavar='PATH',
                       help='模型权重文件路径')
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                       help='分类数量 (默认: 1000)')
    parser.add_argument('--width', type=float, default=1.0, metavar='PCT',
                       help='模型宽度倍数 (默认: 1.0)')
    
    # 数据处理参数
    parser.add_argument('--img-size', type=int, default=224, metavar='N',
                       help='输入图像尺寸 (默认: 224)')
    parser.add_argument('--crop-pct', default=0.875, type=float,
                       metavar='N', help='中心裁剪比例 (默认: 0.875)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                       help='数据集均值')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                       help='数据集标准差')
    parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                       help='图像插值方式 (默认: "bilinear")')
    
    # 测试参数
    parser.add_argument('-b', '--batch-size', type=int, default=256, metavar='N',
                       help='测试批次大小 (默认: 256)')
    parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                       help='数据加载进程数 (默认: 4)')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                       help='固定内存以提高GPU传输效率')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                       help='禁用快速预取器')
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                       help='测试时增强次数 (默认: 0)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                       help='日志输出间隔 (默认: 50)')
    
    # GPU相关
    parser.add_argument('--device', default='cuda', type=str,
                       help='设备 (默认: "cuda")')
    parser.add_argument('--amp', action='store_true', default=False,
                       help='使用混合精度推理')
    
    # 输出相关
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                       help='结果输出路径')
    parser.add_argument('--save-results', action='store_true', default=False,
                       help='保存详细测试结果')
    
    return parser

def load_checkpoint(model, checkpoint_path):
    """加载模型权重"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
    
    logger.info(f'加载权重文件: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理不同格式的权重文件
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 处理权重键名不匹配的情况
    model_state_dict = model.state_dict()
    new_state_dict = {}
    
    for k, v in state_dict.items():
        # 移除 'module.' 前缀（如果存在）
        name = k[7:] if k.startswith('module.') else k
        if name in model_state_dict:
            new_state_dict[name] = v
        else:
            logger.warning(f"权重文件中的键 '{name}' 在模型中不存在")
    
    # 检查缺失的权重
    missing_keys = set(model_state_dict.keys()) - set(new_state_dict.keys())
    if missing_keys:
        logger.warning(f"模型中缺失的权重: {missing_keys}")
    
    model.load_state_dict(new_state_dict, strict=False)
    logger.info('权重加载完成')
    
    return model

def validate(model, loader, args):
    """验证函数"""
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    
    model.eval()
    
    end = time.time()
    last_idx = len(loader) - 1
    results = []
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            
            # 数据移到GPU
            if args.device == 'cuda':
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            
            # 前向传播
            if args.amp:
                with torch.cuda.amp.autocast():
                    output = model(input)
            else:
                output = model(input)
            
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            # TTA处理
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]
            
            # 计算准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            torch.cuda.synchronize()
            
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            
            # 保存详细结果
            if args.save_results:
                pred = output.argmax(dim=1)
                for i in range(len(target)):
                    results.append({
                        'target': target[i].item(),
                        'prediction': pred[i].item(),
                        'correct': (pred[i] == target[i]).item()
                    })
            
            batch_time_m.update(time.time() - end)
            end = time.time()
            
            # 打印进度
            if last_batch or batch_idx % args.log_interval == 0:
                logger.info(
                    f'Test: [{batch_idx:>4d}/{last_idx}] '
                    f'Time: {batch_time_m.val:.3f}s ({batch_time_m.avg:.3f}s) '
                    f'Acc@1: {top1_m.val:>7.4f} ({top1_m.avg:>7.4f}) '
                    f'Acc@5: {top5_m.val:>7.4f} ({top5_m.avg:>7.4f})')
    
    metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])
    
    return metrics, results

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA不可用，使用CPU')
        args.device = 'cpu'
    
    logger.info(f'使用设备: {args.device}')
    
    # 创建模型
    logger.info(f'创建模型: {args.model}')
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        width=args.width,
        dropout=0.0,  # 测试时不使用dropout
        args=args
    )
    
    logger.info(f'模型参数数量: {sum([m.numel() for m in model.parameters()]):,}')
    
    # 加载权重
    model = load_checkpoint(model, args.checkpoint)
    
    # 移动模型到设备
    model = model.to(args.device)
    
    # 解析数据配置
    data_config = resolve_data_config(vars(args), model=model, verbose=True)
    logger.info(f'数据配置: {data_config}')
    
    # 设置数据路径
    eval_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(args.data, 'validation')
        if not os.path.isdir(eval_dir):
            eval_dir = os.path.join(args.data, 'test')
            if not os.path.isdir(eval_dir):
                raise FileNotFoundError(f'找不到验证数据集路径: {args.data}')
    
    logger.info(f'测试数据路径: {eval_dir}')
    
    # 创建数据集
    dataset_eval = Dataset(eval_dir)
    logger.info(f'测试样本数量: {len(dataset_eval)}')
    
    # 创建数据加载器
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=not args.no_prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=False,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )
    
    # 开始测试
    logger.info('开始测试...')
    start_time = time.time()
    
    metrics, results = validate(model, loader_eval, args)
    
    end_time = time.time()
    test_time = end_time - start_time
    
    # 打印结果
    logger.info('=' * 60)
    logger.info('测试完成!')
    logger.info(f'测试时间: {test_time:.2f}秒')
    logger.info(f'样本数量: {len(dataset_eval)}')
    logger.info(f'平均推理时间: {test_time/len(dataset_eval)*1000:.2f}ms/sample')
    logger.info(f'Top-1 准确率: {metrics["top1"]:.4f}%')
    logger.info(f'Top-5 准确率: {metrics["top5"]:.4f}%')
    logger.info('=' * 60)
    
    # 保存结果
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存测试结果摘要
        summary = {
            'model': args.model,
            'checkpoint': args.checkpoint,
            'dataset': args.data,
            'num_samples': len(dataset_eval),
            'batch_size': args.batch_size,
            'test_time': test_time,
            'top1_accuracy': metrics['top1'],
            'top5_accuracy': metrics['top5'],
            'data_config': data_config
        }
        
        summary_path = os.path.join(output_dir, 'test_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        logger.info(f'测试摘要保存至: {summary_path}')
        
        # 保存详细结果
        if args.save_results and results:
            import json
            results_path = os.path.join(output_dir, 'detailed_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f'详细结果保存至: {results_path}')

if __name__ == '__main__':
    main()

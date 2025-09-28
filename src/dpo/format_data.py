#!/usr/bin/env python3
"""
数据处理脚本：将原始数据转换为DPO训练格式

原始数据格式：
{
    "prompt": "...",
    "completion": "...", 
    "reward": 1.0 或 -1.0
}

目标格式：
{
    "chosen": [
        {"content": "prompt", "role": "user"},
        {"content": "completion", "role": "assistant"}
    ],
    "rejected": [
        {"content": "prompt", "role": "user"}, 
        {"content": "completion", "role": "assistant"}
    ],
    "score_chosen": 10,
    "score_rejected": 0
}

python src/dpo/format_data.py --input datasets/aixue_bad_case/all_bad_cases.json --output datasets/aixue_dpo_dataset --format dataset --train_ratio 0.8
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import argparse


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """加载原始数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def group_by_prompt(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按prompt分组数据"""
    grouped = defaultdict(list)
    for item in data:
        prompt = item['prompt']
        grouped[prompt].append(item)
    return grouped


def process_pairs(grouped_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """处理配对数据，生成DPO格式"""
    dpo_data = []
    
    for prompt, items in grouped_data.items():
        if len(items) < 2:
            print(f"警告：prompt只有{len(items)}条数据，跳过")
            continue
            
        # 分离正负reward的数据
        positive_items = [item for item in items if item['reward'] > 0]
        negative_items = [item for item in items if item['reward'] < 0]
        
        if not positive_items or not negative_items:
            print(f"警告：prompt缺少正负样本，跳过")
            continue
            
        # 选择第一个正样本和第一个负样本
        chosen_item = positive_items[0]
        rejected_item = negative_items[0]
        
        # 构建DPO格式数据
        dpo_item = {
            "chosen": [
                {"content": prompt, "role": "user"},
                {"content": chosen_item['completion'], "role": "assistant"}
            ],
            "rejected": [
                {"content": prompt, "role": "user"},
                {"content": rejected_item['completion'], "role": "assistant"}
            ],
            "score_chosen": 10,  # 固定分数
            "score_rejected": 0  # 固定分数
        }
        
        dpo_data.append(dpo_item)
    
    return dpo_data


def create_dataset_structure(data: List[Dict[str, Any]], output_dir: str, train_ratio: float = 0.8):
    """创建标准的数据集目录结构，支持load_dataset加载"""
    output_path = Path(output_dir)
    
    # 创建目录结构
    data_dir = output_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 分割训练和测试数据
    import random
    random.seed(42)  # 固定随机种子确保可重复
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # 保存训练数据
    train_df = pd.DataFrame(train_data)
    train_file = data_dir / "train-00000-of-00001.parquet"
    train_df.to_parquet(train_file, index=False)
    
    # 保存测试数据
    test_df = pd.DataFrame(test_data)
    test_file = data_dir / "test-00000-of-00001.parquet"
    test_df.to_parquet(test_file, index=False)
    
    # 创建README.md文件
    readme_content = f"""---
dataset_info:
  features:
  - name: chosen
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: rejected
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: score_chosen
    dtype: float64
  - name: score_rejected
    dtype: float64
  splits:
  - name: train
    num_bytes: {train_file.stat().st_size}
    num_examples: {len(train_data)}
  - name: test
    num_bytes: {test_file.stat().st_size}
    num_examples: {len(test_data)}
  download_size: {train_file.stat().st_size + test_file.stat().st_size}
  dataset_size: {train_file.stat().st_size + test_file.stat().st_size}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
"""
    
    readme_file = output_path / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"已创建数据集结构：")
    print(f"  训练数据: {len(train_data)} 条 -> {train_file}")
    print(f"  测试数据: {len(test_data)} 条 -> {test_file}")
    print(f"  数据集信息: {readme_file}")
    print(f"  使用方式: load_dataset('{output_dir}')")


def save_parquet(data: List[Dict[str, Any]], output_path: str):
    """保存为parquet格式"""
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)
    print(f"已保存 {len(data)} 条数据到 {output_path}")


def save_json(data: List[Dict[str, Any]], output_path: str):
    """保存为JSON格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(data)} 条数据到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description='将原始数据转换为DPO训练格式')
    parser.add_argument('--input', '-i', required=True, help='输入JSON文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出文件路径或目录')
    parser.add_argument('--format', '-f', choices=['parquet', 'json', 'dataset'], default='dataset', 
                       help='输出格式 (默认: dataset，支持load_dataset加载)')
    parser.add_argument('--train_ratio', '-r', type=float, default=0.8, 
                       help='训练集比例 (默认: 0.8)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：输入文件 {input_path} 不存在")
        return
    
    # 加载数据
    print(f"加载数据：{input_path}")
    data = load_data(str(input_path))
    print(f"原始数据条数：{len(data)}")
    
    # 按prompt分组
    grouped_data = group_by_prompt(data)
    print(f"唯一prompt数量：{len(grouped_data)}")
    
    # 处理配对数据
    dpo_data = process_pairs(grouped_data)
    print(f"生成DPO数据条数：{len(dpo_data)}")
    
    # 保存数据
    if args.format == 'dataset':
        # 创建标准数据集结构
        create_dataset_structure(dpo_data, args.output, args.train_ratio)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'parquet':
            save_parquet(dpo_data, str(output_path))
        else:
            save_json(dpo_data, str(output_path))
    
    print("数据处理完成！")


if __name__ == "__main__":
    main()

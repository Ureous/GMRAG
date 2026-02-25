"""
通用工具函数模块
用于项目中的数据处理、文件操作等常用功能
"""
import json
import random
import torch
import numpy as np
from typing import Any, Dict, List, Union


def set_seed(seed: int = 42):
    """
    设置随机种子，确保结果可复现
    
    Args:
        seed: 随机种子值，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_json(file_path: str) -> Union[Dict, List]:
    """
    读取JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        JSON文件内容（字典或列表）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(file_path: str) -> List[Dict]:
    """
    读取JSONL文件（每行一个JSON对象）
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        包含所有JSON对象的列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_json(data: Union[Dict, List], file_path: str):
    """
    写入JSON文件
    
    Args:
        data: 要写入的数据（字典或列表）
        file_path: 输出文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_to_json_file(data: Union[Dict, List], file_path: str):
    """
    追加数据到JSONL文件（每行一个JSON对象）
    
    Args:
        data: 要追加的数据（字典或列表）
        file_path: JSONL文件路径
    """
    with open(file_path, 'a', encoding='utf-8') as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def write_jsonl(data: List[Dict], file_path: str):
    """
    写入JSONL文件（覆盖模式）
    
    Args:
        data: 要写入的数据列表
        file_path: 输出文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# 为了向后兼容，添加常用的别名
load_json = read_json
save_json = write_json

# src/data/datasets_reft.py
"""
ReftClassificationDataset 的原始设计是为了处理文本到文本的分类任务（例如，输入是问题，标签也是文本答案），而不是文本到类别的标准分类任务，因此我们自定义一个新的用于整数label的分类数据集
"""
from pyreft.dataset import ReftClassificationDataset, ReftDataCollator, ReftDataset
from sklearn import base
from transformers import DataCollatorWithPadding, AutoTokenizer
from typing import List, Optional
import torch
from torch.utils.data import Subset
from datasets import load_dataset

class ReftClassificationDatasetForIntLabels(ReftDataset):
    def load_dataset(self):
        """
        修正 1：方法名必须是 load_dataset，且不带额外参数。
        """
        # 如果 self.dataset 已经由构造函数传入（即我们之前的 memory_dataset 方案）
        if self.dataset is not None:
            return self.dataset
            
        # 否则根据 task 类型加载
        if self.task == "json" or (self.data_path and self.data_path.endswith(".json")):
            ds = load_dataset("json", data_files=self.data_path)
            return ds[self.data_split]
            
        # 否则回退到父类的 Hub 加载逻辑
        # 注意：父类的方法名是 load_dataset 而不是 load_data
        return super().load_dataset()
        
    def preprocess(self, kwargs):
        super().preprocess(kwargs)
        self.input_field = kwargs["input_field"]
        self.label_field = kwargs["label_field"]
        
    def tokenize(self, data_item):
        max_len = min(self.tokenizer.model_max_length, 512)
        result = {}
        enc = self.tokenizer(data_item[self.input_field], max_length=max_len,
            truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]        
        #pdb.set_trace()
        base_prompt_length = len(input_ids)
        # last_position = base_prompt_length - 1 # 干预倒数第二个token，因为采用了新的compute_intervention_and_subspaces，没有在开头加padding
        last_position = base_prompt_length # 保证干预最后一个token
        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask

        # labels: 直接使用整数label 不做tokenizer
        # 需要转化为torch.Tensor
        result["labels"] = torch.tensor([int(data_item[self.label_field])], dtype=torch.long) 
        return result, last_position
    
    def compute_intervention_and_subspaces(self, id, data_item, result, last_position, **kwargs):
        """
        覆盖父类方法，计算 intervention locations，因为我们的标签是整数
        1. 根据last_position和num_interventions计算intervention位置
        2. 记录id
        不再对 input_ids / labels 做额外处理
        attack token 和 reft token对应，相同
        """
        intervention_locations = self.get_intervention_locations(last_position=last_position, first_n=self.first_n, 
            last_n=self.last_n, pad_mode=self.pad_mode, **kwargs)
        result["intervention_locations"] = torch.tensor(intervention_locations, dtype=torch.int)
        result["id"] = id
        return result
    

def build_reft_classification_datasets(
    tokenizer: AutoTokenizer,
    data_path: str = "AlignmentResearch/Harmless",
    train_split: str = "train",
    eval_split: str = "validation",
    train_indices: Optional[List[int]] = None, # 支持传入索引
    input_field: str = "content",
    label_field: str = "clf_label",
    position: str = "l1",
    num_interventions: int = 2,
    min_seq_length: int = 100,  # 新增参数：验证集最小长度过滤
):
    """
    构建基于pyreft的datasets
    """
    # 先按标准方式构建ReftClassificationDataset
    # 加载完整的train split
     
    dataset_repo = data_path
    dataset_config = None
    print(f"[datasets] 正在从 {dataset_repo} (split='{train_split}') 加载 ReFT 训练集...")
    full_train_dataset = ReftClassificationDatasetForIntLabels(
        task=dataset_repo,
        data_path=dataset_config,
        tokenizer=tokenizer,
        data_split=train_split,
        position=position,
        num_interventions=num_interventions,
        input_field=input_field,
        label_field=label_field,
    )
        
    
    print(f"[datasets] 正在从 {dataset_repo} (split='{eval_split}') 加载 ReFT 验证集...")
    full_eval_dataset = ReftClassificationDatasetForIntLabels(
        task=dataset_repo,
        data_path=dataset_config,
        tokenizer=tokenizer,
        data_split=eval_split,
        position=position,
        num_interventions=num_interventions,
        input_field=input_field,
        label_field=label_field,
    )
    
    # 1. 训练集：先应用 train_indices 切割成基础子集 (如果有)
    if train_indices is not None:
        print(f"[datasets] 原始 ReFT 训练集 {len(full_train_dataset)}, 选择 {len(train_indices)} 条索引子集")
        base_train_dataset = Subset(full_train_dataset, train_indices)
    else:
        print(f"[datasets] 使用完整的 ReFT 训练集 ({len(full_train_dataset)} 条)")
        base_train_dataset = full_train_dataset

    # 2. 定义通用的长度过滤函数
    def filter_dataset_by_length(dataset, min_len, desc="Filtering"):
        if min_len <= 0:
            return dataset
            
        from tqdm import tqdm
        valid_indices = []
        print(f"[datasets] 正在清洗 {desc} (剔除 token 长度 < {min_len} 的样本)...")
        
        for i in tqdm(range(len(dataset)), desc=desc):
            item = dataset[i]
            # attention_mask 求和就是实际的 token 数量 (不包含 padding)
            seq_len = item["attention_mask"].sum().item()
            
            if seq_len >= min_len:
                valid_indices.append(i)
                
        filtered_ds = Subset(dataset, valid_indices)
        print(f"[datasets] {desc} 清洗完毕: 从 {len(dataset)} 缩减到 {len(filtered_ds)} 条。")
        return filtered_ds

    # 3. 对训练集和验证集分别进行长度过滤
    train_dataset = filter_dataset_by_length(base_train_dataset, min_seq_length, desc="Train Set")
    eval_dataset = filter_dataset_by_length(full_eval_dataset, min_seq_length, desc="Eval Set")

    return train_dataset, eval_dataset


def build_reft_data_collator(tokenizer, max_length=512):
    """构建 pyreft 数据整理器"""
    hf_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return ReftDataCollator(data_collator=hf_collator)

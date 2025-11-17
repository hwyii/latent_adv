"""
ReftClassificationDataset 的原始设计是为了处理文本到文本的分类任务（例如，输入是问题，标签也是文本答案），而不是文本到类别的标准分类任务，因此我们自定义一个新的用于整数label的分类数据集
"""
import pyreft
from pyreft.dataset import ReftClassificationDataset, ReftDataCollator, ReftDataset
from transformers import DataCollatorWithPadding, AutoTokenizer
from typing import List, Optional
import torch
from torch.utils.data import Subset
from datasets import load_dataset

class ReftClassificationDatasetForIntLabels(ReftDataset):
    
    def preprocess(self, kwargs):
        super().preprocess(kwargs)
        self.input_field = kwargs["input_field"]
        self.label_field = kwargs["label_field"]
        
    def tokenize(self, data_item):

        result = {}
        input_ids = self.tokenizer(data_item[self.input_field], max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")["input_ids"][0]
    
        base_prompt_length = len(input_ids)
        last_position = base_prompt_length - 1
        result["input_ids"] = input_ids

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
    eval_dataset = ReftClassificationDatasetForIntLabels(
        task=dataset_repo,
        data_path=dataset_config,
        tokenizer=tokenizer,
        data_split=eval_split,
        position=position,
        num_interventions=num_interventions,
        input_field=input_field,
        label_field=label_field,
    )
    
    if train_indices is not None:
        print(f"[datasets] 原始 ReFT 训练集 {len(full_train_dataset)}, 选择 {len(train_indices)} 条索引子集")
        train_dataset = Subset(full_train_dataset, train_indices)
    else:
        print(f"[datasets] 使用完整的 ReFT 训练集 ({len(full_train_dataset)} 条)")
        train_dataset = full_train_dataset

    # 在返回之前添加    
    sample = train_dataset[0] if not isinstance(train_dataset, Subset) else train_dataset.dataset[train_dataset.indices[0]]  
    # print("[DEBUG] sample keys:", sample.keys())
    # print("[DEBUG] input_ids.shape:", sample["input_ids"].shape)
    # print("[DEBUG] labels:", sample["labels"])
    # print(f"  intervention_locations: {sample['intervention_locations']}")  
    # print(f"  input_ids 长度: {len(sample['input_ids'])}")
    return train_dataset, eval_dataset


def build_reft_data_collator(tokenizer, max_length=512):
    """构建 pyreft 数据整理器"""
    hf_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
    )
    return ReftDataCollator(data_collator=hf_collator)

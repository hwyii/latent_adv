# enron_subset.py
from torch.utils.data import Dataset
from datasets import load_dataset

class AdvDataset(Dataset):
    def __init__(self, split_indices=None, dataset_name=None, split:str="train"):
        """
        split_indices: list[int]
        """
        ds = load_dataset(dataset_name)[split]
        self._base = ds
        if split_indices is None:
            self._indices = list(range(len(ds)))
        else:
            self._indices = list(split_indices)

    def __len__(self): return len(self._indices)

    def __getitem__(self, i):
        j = self._indices[i]
        item = self._base[j]
        text = item['content'][0] if isinstance(item['content'], list) else item['content']
        label = int(item['clf_label']) 
        return {"text": text, "label": label}  


# check dataset

# import json
# with open("enron_splits.json", "r") as f:
#     splits = json.load(f)
# probe_ds = AdvDataset(splits["probe"])
# train_ds = AdvDataset(splits["ft_train"])
# val_ds = AdvDataset(splits["ft_val"])
# attack_ds = AdvDataset(splits["attack"])
# print("Probe 数据集样本数:", len(probe_ds))
# print("第 0 条数据：")
# print(probe_ds[0])

# print("train 数据集样本数:", len(train_ds))
# print("第 0 条数据：")
# print(train_ds[0])

# print("val 数据集样本数:", len(val_ds))
# print("第 0 条数据：")
# print(val_ds[0])

# print("attack 数据集样本数:", len(attack_ds))
# print("第 0 条数据：")
# print(attack_ds[0])

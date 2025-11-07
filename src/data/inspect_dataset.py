# scripts/inspect_enron_spam.py
from cProfile import label
from datasets import load_dataset
from collections import Counter

def inspect_data(dataset_path=None):
    ds = load_dataset(dataset_path)   # 默认会拿到 train/validation
    print(ds)                                          # 看 splits 概览

    train = ds["validation"]
    print(len(train))                 # 看字段/类型（ClassLabel等）


    # row = train[0]
    # print("\nA sample row:", row)                      # 看一行数据
    # text = row["content"][0] if isinstance(row["content"], list) else row["content"]
    # label = row["clf_label"]                       # int: 0 or 1
    # #print(f"\n--- sample {0} ---")
    # #print("text[:300]:", text[:300].replace("\n"," "))
    # print("label:", label)

    # 标签分布
    cnt = Counter(int(r["clf_label"]) for r in train)
    print("\nLabel histogram (train):", cnt)
inspect_data("AlignmentResearch/Helpful")



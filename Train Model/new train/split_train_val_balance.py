import json
import random
from collections import defaultdict

# 配置
input_file      = "../self train/train_index.json"
train_out_file  = "train_split_balanced.json"
val_out_file    = "val_split_balanced.json"

val_ratio    = 0.1    # 验证集占比
random_seed  = 42     # 随机种子，保证可复现

random.seed(random_seed)

# 1. 读取所有样本
with open(input_file, 'r', encoding='utf-8') as f:
    all_samples = json.load(f)

# 2. 按标签分组
by_label = defaultdict(list)
for item in all_samples:
    by_label[item["label"]].append(item)

# 3. 对每个标签内部打乱，并预先划分出验证集候选
val_candidates = {}
train_candidates = {}
for label, samples in by_label.items():
    random.shuffle(samples)
    n_val = int(len(samples) * val_ratio)
    val_candidates[label] = samples[:n_val]
    train_candidates[label] = samples[n_val:]

# 4. 确定各标签下验证集和平衡后训练集的样本数
min_val_count   = min(len(v) for v in val_candidates.values())
min_train_count = min(len(v) for v in train_candidates.values())

# 5. 从每个标签中截取相同数量，构建最终验证集和训练集
final_val = []
final_train = []
for label in by_label:
    final_val   .extend(val_candidates[label][:min_val_count])
    final_train .extend(train_candidates[label][:min_train_count])

# 6. 打乱并写入文件
random.shuffle(final_val)
random.shuffle(final_train)
with open(val_out_file, 'w', encoding='utf-8') as f:
    json.dump(final_val, f, ensure_ascii=False, indent=2)
with open(train_out_file, 'w', encoding='utf-8') as f:
    json.dump(final_train, f, ensure_ascii=False, indent=2)

print(f"✅ Saved balanced splits: {len(final_train)} train ({min_train_count}×labels), {len(final_val)} val ({min_val_count}×labels)")

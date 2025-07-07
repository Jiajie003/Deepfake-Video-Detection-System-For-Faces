import json
import random
from collections import defaultdict

input_json = "train_index.json"
train_output = "train_split.json"
val_output = "val_split.json"
val_ratio = 0.1  # 10% for validation

# Load all data
with open(input_json, 'r') as f:
    data = json.load(f)

# Group by label
label_groups = defaultdict(list)
for entry in data:
    label_groups[entry["label"]].append(entry)

# Stratified split
train_set = []
val_set = []

for label, samples in label_groups.items():
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_ratio))
    train_set.extend(samples[:split_idx])
    val_set.extend(samples[split_idx:])

# Shuffle both splits
random.shuffle(train_set)
random.shuffle(val_set)

# Save
with open(train_output, 'w') as f:
    json.dump(train_set, f, indent=2)

with open(val_output, 'w') as f:
    json.dump(val_set, f, indent=2)

print(f"✅ Total: {len(data)}")
print(f"📦 Training: {len(train_set)} samples → {train_output}")
print(f"🧪 Validation: {len(val_set)} samples → {val_output}")

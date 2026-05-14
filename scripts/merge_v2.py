#!/usr/bin/env python3
"""
Merge all data sources, deduplicate, split, and convert to training format.
v2: 1081+ items, stratified split, training format (instruction/input/output/system/category).
"""

import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(789)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load and deduplicate all sources
all_items = []
seen = set()

sources = [
    'cleaned_data.jsonl',
    'diverse_generated.jsonl',
    'diverse_v2.jsonl',
]

for fname in sources:
    fpath = DATA_DIR / fname
    if fpath.exists():
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    key = item['optimized'].strip().lower()[:150]
                    if key not in seen:
                        seen.add(key)
                        all_items.append(item)

print(f"Total unique items: {len(all_items)}")

# Validate
valid = []
for item in all_items:
    out = item['optimized'].strip()
    # Remove any remaining prefixes
    for prefix in ["Optimized Prompt:", "Optimized prompt:", "**Optimized:**",
                    "**Optimized Prompt:**"]:
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
    if out.startswith("**"):
        out = out.lstrip("*").lstrip(":").strip()
    
    # Validate
    if len(out) < 40 or len(out) > 600:
        continue
    if any(a in out.lower() for a in ["use none", "-/-/-"]):
        continue
    if out.lower().startswith(("optimized prompt", "here is", "here's")):
        continue
    
    item['optimized'] = out
    valid.append(item)

print(f"Valid items after cleanup: {len(valid)}")

# Stratified split: 80/10/10 by category
train, val, test = [], [], []
cat_items = defaultdict(list)
for item in valid:
    cat_items[item['category']].append(item)

for cat, items in cat_items.items():
    random.shuffle(items)
    n = len(items)
    n_test = max(1, round(n * 0.10))
    n_val = max(1, round(n * 0.10))
    n_train = n - n_test - n_val
    
    train.extend(items[:n_train])
    val.extend(items[n_train:n_train + n_val])
    test.extend(items[n_train + n_val:])

# Shuffle each split
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

# Convert to training format
SYSTEM_PROMPT = "You are a prompt engineering expert. Transform vague, underspecified prompts into clear, specific, and effective ones. Preserve the original intent while adding concrete details, structure, and actionable guidance. Output only the optimized prompt — no prefixes, no labels, no explanations."

INSTRUCTION = "Optimize the following prompt to be clear, specific, and effective. Preserve the original intent while adding concrete details, structure, and actionable guidance."

def to_training_format(item):
    return {
        "instruction": INSTRUCTION,
        "input": item["vague"],
        "output": item["optimized"],
        "system": SYSTEM_PROMPT,
        "category": item["category"],
    }

train_fmt = [to_training_format(item) for item in train]
val_fmt = [to_training_format(item) for item in val]
test_fmt = [to_training_format(item) for item in test]

# Save JSONL
for name, data in [("train", train_fmt), ("val", val_fmt), ("test", test_fmt)]:
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {name}.jsonl: {len(data)} items")

# Save combined dataset
all_fmt = train_fmt + val_fmt + test_fmt
with open(DATA_DIR / "final_dataset.jsonl", "w") as f:
    for item in all_fmt:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved final_dataset.jsonl: {len(all_fmt)} items")

# Save split metadata
split_info = {
    "total": len(all_fmt),
    "train": len(train_fmt),
    "val": len(val_fmt),
    "test": len(test_fmt),
    "categories": {cat: len(items) for cat, items in cat_items.items()},
    "system_prompt": SYSTEM_PROMPT,
    "instruction": INSTRUCTION,
}
with open(DATA_DIR / "split.json", "w") as f:
    json.dump(split_info, f, indent=2, ensure_ascii=False)
print(f"\nMetadata saved to split.json")
print(f"\nCategory distribution:")
for cat in sorted(split_info["categories"]):
    print(f"  {cat}: {split_info['categories'][cat]}")
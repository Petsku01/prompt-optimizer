#!/usr/bin/env python3
"""
Split cleaned data into train/val/test sets with category balancing.

Splits: 80% train, 10% val, 10% test
Ensures each category has proportional representation in all splits.
"""

import json
import random
from pathlib import Path
from collections import defaultdict


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: Path):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_to_instruction_format(item: dict) -> dict:
    """Convert to standard instruction-tuning format for Qwen2.5."""
    return {
        "instruction": "Optimize the following prompt to be clear, specific, and effective. Preserve the original intent while adding structure, context, and constraints where appropriate.",
        "input": item["vague"],
        "output": item["optimized"],
        "category": item.get("category", "unknown"),
        "system": "You are a prompt engineering expert that transforms vague, underspecified prompts into clear, well-structured, and effective prompts. You preserve the user's original intent while adding relevant specificity, format constraints, audience targeting, and contextual details."
    }


def split_data(items: list[dict], train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split data ensuring category balance across all sets."""
    random.seed(seed)

    by_category = defaultdict(list)
    for item in items:
        cat = item.get("category", "unknown")
        by_category[cat].append(item)

    train, val, test = [], [], []

    for cat, cat_items in by_category.items():
        random.shuffle(cat_items)
        n = len(cat_items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train.extend(cat_items[:n_train])
        val.extend(cat_items[n_train:n_train + n_val])
        test.extend(cat_items[n_train + n_val:])

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def main():
    data_dir = Path(__file__).parent.parent / "data"
    cleaned_path = data_dir / "cleaned_data.jsonl"

    if not cleaned_path.exists():
        print(f"Error: {cleaned_path} not found. Run clean_data.py first.")
        return

    items = load_jsonl(cleaned_path)
    print(f"Loaded {len(items)} cleaned items")

    # Convert to instruction format
    formatted = [convert_to_instruction_format(item) for item in items]

    # Split
    train, val, test = split_data(formatted)

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Save
    save_jsonl(train, data_dir / "train.jsonl")
    save_jsonl(val, data_dir / "val.jsonl")
    save_jsonl(test, data_dir / "test.jsonl")

    # Print category distribution per split
    from collections import Counter
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        cats = Counter(i.get("category", "unknown") for i in split)
        print(f"\n{name} categories:")
        for cat, count in cats.most_common():
            print(f"  {cat}: {count}")

    # Also save as JSON for easy inspection
    for name, split in [("train", train), ("val", val), ("test", test)]:
        json_path = data_dir / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Files saved to {data_dir}")


if __name__ == "__main__":
    main()
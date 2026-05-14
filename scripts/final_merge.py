#!/usr/bin/env python3
"""
Final dataset merge, clean, and split pipeline.
Combines diverse generated data with cleaned legacy data.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MIN_CATEGORY_SIZE = 15


def normalize(s: str) -> str:
    return ' '.join(s.lower().split())


def main():
    # Load diverse generated data
    diverse_path = DATA_DIR / "diverse_generated.jsonl"
    with open(diverse_path) as f:
        diverse = [json.loads(l) for l in f if l.strip()]
    print(f"Diverse generated: {len(diverse)} items")

    # Load clean legacy data
    clean_path = DATA_DIR / "cleaned_data.jsonl"
    with open(clean_path) as f:
        clean = [json.loads(l) for l in f if l.strip()]
    print(f"Clean legacy: {len(clean)} items")

    # Combine, preferring diverse data for duplicates
    all_items = []
    seen_outputs = set()

    # Add diverse first (higher quality)
    for item in diverse:
        key = normalize(item["optimized"])
        if key not in seen_outputs:
            seen_outputs.add(key)
            all_items.append(item)

    # Add clean legacy items that don't overlap
    for item in clean:
        key = normalize(item["optimized"])
        if key not in seen_outputs:
            seen_outputs.add(key)
            all_items.append(item)

    print(f"After dedup: {len(all_items)} items")

    # Category counts
    cat_counts = defaultdict(int)
    for item in all_items:
        cat_counts[item["category"]] += 1

    print("\nBefore category filtering:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    # Remove categories below minimum
    valid_cats = {cat for cat, cnt in cat_counts.items() if cnt >= MIN_CATEGORY_SIZE}
    all_items = [item for item in all_items if item["category"] in valid_cats]
    print(f"\nAfter removing categories < {MIN_CATEGORY_SIZE}: {len(all_items)} items")

    # Final category counts
    cat_counts = defaultdict(int)
    for item in all_items:
        cat_counts[item["category"]] += 1

    print("\nFinal category distribution:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    # Stratified split by category
    train, val, test = [], [], []

    for cat in valid_cats:
        cat_items = [item for item in all_items if item["category"] == cat]
        random.shuffle(cat_items)
        n = len(cat_items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        # Ensure at least 2 in val and 2 in test for small categories
        if n >= 15:
            n_val = max(2, n_val)
            n_test = max(2, n - n_train - n_val)
        else:
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)

        n_train = n - n_val - n_test

        train.extend(cat_items[:n_train])
        val.extend(cat_items[n_train:n_train + n_val])
        test.extend(cat_items[n_train + n_val:n_train + n_val + n_test])

    # Shuffle splits
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    # Verify no leakage
    train_outputs = set(normalize(item["optimized"]) for item in train)
    val_outputs = set(normalize(item["optimized"]) for item in val)
    test_outputs = set(normalize(item["optimized"]) for item in test)

    print(f"Output leakage: train∩val={len(train_outputs & val_outputs)}, train∩test={len(train_outputs & test_outputs)}, val∩test={len(val_outputs & test_outputs)}")

    # Convert to training format
    SYSTEM_PROMPT = "You are a prompt engineering expert that transforms vague, underspecified prompts into clear, well-structured, and effective prompts. You preserve the user's original intent while adding relevant specificity, format constraints, audience targeting, and contextual details."
    INSTRUCTION = "Optimize the following prompt to be clear, specific, and effective. Preserve the original intent while adding structure, context, and constraints where appropriate."

    def to_training(item):
        return {
            "instruction": INSTRUCTION,
            "input": item["vague"],
            "output": item["optimized"],
            "category": item["category"],
            "system": SYSTEM_PROMPT,
        }

    # Save all formats
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        # JSONL (source format)
        jsonl_path = DATA_DIR / f"{split_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # JSON (training format)
        json_path = DATA_DIR / f"{split_name}.json"
        formatted = [to_training(item) for item in split_data]
        with open(json_path, "w") as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)

        print(f"  {split_name}: {len(split_data)} items → {jsonl_path.name} + {json_path.name}")

    # Save combined cleaned data
    all_sorted = train + val + test
    with open(DATA_DIR / "cleaned_data.jsonl", "w") as f:
        for item in all_sorted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_sorted)} items saved to cleaned_data.jsonl")

    # Print final stats
    print("\n" + "=" * 50)
    print("FINAL DATASET SUMMARY")
    print("=" * 50)
    final_cats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for item in train:
        final_cats[item["category"]]["train"] += 1
    for item in val:
        final_cats[item["category"]]["val"] += 1
    for item in test:
        final_cats[item["category"]]["test"] += 1

    print(f"{'Category':<15} {'Train':>6} {'Val':>4} {'Test':>5} {'Total':>6}")
    print("-" * 40)
    for cat in sorted(final_cats.keys()):
        c = final_cats[cat]
        total = c["train"] + c["val"] + c["test"]
        print(f"{cat:<15} {c['train']:>6} {c['val']:>4} {c['test']:>5} {total:>6}")
    print("-" * 40)
    print(f"{'TOTAL':<15} {len(train):>6} {len(val):>4} {len(test):>5} {len(all_sorted):>6}")

    # Output length stats
    print(f"\n{'Category':<15} {'Min':>4} {'Mean':>5} {'Max':>4}")
    print("-" * 30)
    for cat in sorted(final_cats.keys()):
        lens = sorted([len(item["optimized"]) for item in all_sorted if item["category"] == cat])
        print(f"{cat:<15} {min(lens):>4} {sum(lens)//len(lens):>5} {max(lens):>4}")


if __name__ == "__main__":
    main()
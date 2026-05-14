#!/usr/bin/env python3
"""Clean and regenerate the prompt optimizer dataset.

Fixes:
1. Remove near-duplicate inputs (Jaccard similarity > 0.8)
2. Remove 'mixed' category (too few samples)
3. Remove under-developed writing outputs (< 120 chars)
4. Regenerate stratified 80/10/10 splits
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MIN_OUTPUT_LEN_WRITING = 120
DROP_CATEGORIES = {"mixed"}
JACCARD_THRESHOLD = 0.8

BASE_DIR = Path(__file__).resolve().parent.parent


def word_set(s: str) -> set:
    return set(s.lower().split())


def jaccard(s1: str, s2: str) -> float:
    w1, w2 = word_set(s1), word_set(s2)
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def deduplicate(data: list[dict]) -> list[dict]:
    """Remove near-duplicate inputs, keeping the one with more detailed output."""
    # Sort by output length descending so we prefer more specific outputs
    data_sorted = sorted(data, key=lambda x: len(x["optimized"]), reverse=True)

    to_remove: set[int] = set()
    for i in range(len(data_sorted)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(data_sorted)):
            if j in to_remove:
                continue
            if jaccard(data_sorted[i]["vague"], data_sorted[j]["vague"]) > JACCARD_THRESHOLD:
                to_remove.add(j)

    return [item for idx, item in enumerate(data_sorted) if idx not in to_remove]


def filter_quality(data: list[dict]) -> list[dict]:
    """Remove under-developed writing outputs and rare categories."""
    filtered = []
    for item in data:
        # Drop rare categories
        if item["category"] in DROP_CATEGORIES:
            continue
        # Drop short writing outputs
        if item["category"] == "writing" and len(item["optimized"]) < MIN_OUTPUT_LEN_WRITING:
            continue
        filtered.append(item)
    return filtered


def stratified_split(data: list[dict], train_ratio: float, val_ratio: float, seed: int):
    """Split data stratified by category."""
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for item in data:
        by_category[item["category"]].append(item)

    train, val, test = [], [], []
    for cat, items in by_category.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Ensure at least 1 in test
        if n - n_train - n_val < 1:
            n_val = max(0, n - n_train - 1)

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    return train, val, test


def to_training_format(item: dict) -> dict:
    """Convert from (vague, optimized, category) to the training format."""
    return {
        "instruction": "Optimize the following prompt to be clear, specific, and effective. Preserve the original intent while adding structure, context, and constraints where appropriate.",
        "input": item["vague"],
        "output": item["optimized"],
        "category": item["category"],
        "system": "You are a prompt engineering expert that transforms vague, underspecified prompts into clear, well-structured, and effective prompts. You preserve the user's original intent while adding relevant specificity, format constraints, audience targeting, and contextual details.",
    }


def main():
    random.seed(SEED)

    # Load source data
    src = BASE_DIR / "data" / "cleaned_data.jsonl"
    with open(src) as f:
        data = [json.loads(line) for line in f]

    print(f"Source: {len(data)} items")

    # Step 1: Deduplicate
    deduped = deduplicate(data)
    print(f"After dedup: {len(deduped)} items (removed {len(data) - len(deduped)})")

    # Step 2: Quality filter
    filtered = filter_quality(deduped)
    print(f"After quality filter: {len(filtered)} items (removed {len(deduped) - len(filtered)})")

    # Step 3: Stratified split
    train, val, test = stratified_split(filtered, TRAIN_RATIO, VAL_RATIO, SEED)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Step 4: Convert and save
    out_dir = BASE_DIR / "data"
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for item in split_data:
                f.write(json.dumps(to_training_format(item), ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_data)} items → {out_path.relative_to(BASE_DIR)}")

    # Also save train/val/test as JSON (for Colab)
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out_path = out_dir / f"{split_name}.json"
        formatted = [to_training_format(item) for item in split_data]
        with open(out_path, "w") as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n=== Category distribution (train) ===")
    for cat in sorted(set(item["category"] for item in train)):
        cnt = sum(1 for item in train if item["category"] == cat)
        print(f"  {cat}: {cnt}")

    print(f"\n=== Output length stats (train) ===")
    for cat in sorted(set(item["category"] for item in train)):
        lens = sorted([len(item["optimized"]) for item in train if item["category"] == cat])
        if lens:
            print(f"  {cat}: mean={sum(lens)/len(lens):.0f}, med={lens[len(lens)//2]}, min={min(lens)}, max={max(lens)}")


if __name__ == "__main__":
    main()
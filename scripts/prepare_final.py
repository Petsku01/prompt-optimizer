#!/usr/bin/env python3
"""
Quick pipeline: augment + clean + split with better dedup threshold.
"""

import json
import random
import re
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items, path):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def validate_pair(item):
    vague = item.get("vague", "")
    optimized = item.get("optimized", "")
    if not vague or not optimized:
        return False
    if len(vague.strip()) < 2:
        return False
    if len(optimized.strip()) < 20:
        return False
    if len(optimized) > 3000:
        return False
    if len(optimized) < len(vague) * 1.2:
        return False
    if optimized.strip().lower() == vague.strip().lower():
        return False
    return True


def pair_hash(item):
    """Create a hash of the (vague, optimized) pair for exact dedup."""
    return hash((item["vague"].lower().strip(), item["optimized"].lower().strip()[:50]))


def main():
    data_dir = Path(__file__).parent.parent / "data"
    raw_dir = data_dir / "raw"

    # Load all data
    all_items = []
    for fname in ["seed_data.jsonl", "generated_data.jsonl", "augmented_data.jsonl"]:
        path = raw_dir / fname
        if path.exists():
            items = load_jsonl(path)
            print(f"Loaded {len(items)} from {fname}")
            all_items.extend(items)

    print(f"\nTotal raw: {len(all_items)}")

    # Validate
    validated = [i for i in all_items if validate_pair(i)]
    print(f"After validation: {len(validated)}")

    # Exact dedup of (vague, optimized) pairs
    seen_hashes = set()
    unique = []
    for item in validated:
        h = pair_hash(item)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(item)
    print(f"After exact pair dedup: {len(unique)}")

    # Near-dup of vague only (more lenient threshold - only remove if VERY similar)
    seen_vague = []
    kept = []
    pairs_by_vague = {}
    for item in unique:
        vague_key = item["vague"].lower().strip()
        if vague_key not in pairs_by_vague:
            pairs_by_vague[vague_key] = []
        pairs_by_vague[vague_key].append(item)

    # Keep all unique vague prompts that differ enough
    for vague_key, items in pairs_by_vague.items():
        # Keep the best optimized version for each vague
        best = max(items, key=lambda x: len(x["optimized"]))
        kept.append(best)

    print(f"After vague-level dedup (keep best): {len(kept)}")

    # Category stats
    cat_counts = Counter(i.get("category", "unknown") for i in kept)
    print("\nCategory distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    # Balance: cap over-represented categories
    cat_limits = {
        "writing": 200, "coding": 200, "analysis": 200, "q_and_a": 200,
        "roleplay": 150, "summarization": 150, "brainstorming": 150,
        "translation": 150, "instruction": 100, "editing": 100, "mixed": 10,
    }

    balanced = []
    cat_seen = Counter()
    random.seed(42)
    random.shuffle(kept)  # Randomize before capping
    for item in kept:
        cat = item.get("category", "unknown")
        limit = cat_limits.get(cat, 150)
        if cat_seen[cat] < limit:
            balanced.append(item)
            cat_seen[cat] += 1

    print(f"\nAfter balancing: {len(balanced)}")
    for cat, count in cat_seen.most_common():
        print(f"  {cat}: {count}")

    # Split: 80/10/10
    random.shuffle(balanced)
    n = len(balanced)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train = balanced[:train_end]
    val = balanced[train_end:val_end]
    test = balanced[val_end:]

    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    # Convert to instruction format
    def to_instruction(item):
        return {
            "instruction": "Optimize the following prompt to be clear, specific, and effective. Preserve the original intent while adding structure, context, and constraints where appropriate.",
            "input": item["vague"],
            "output": item["optimized"],
            "category": item.get("category", "unknown"),
            "system": "You are a prompt engineering expert that transforms vague, underspecified prompts into clear, well-structured, and effective prompts. You preserve the user's original intent while adding relevant specificity, format constraints, audience targeting, and contextual details."
        }

    # Save all splits
    for name, split in [("train", train), ("val", val), ("test", test)]:
        formatted = [to_instruction(item) for item in split]
        save_jsonl(formatted, data_dir / f"{name}.jsonl")
        # Also save as JSON for easy inspection
        with open(data_dir / f"{name}.json", "w") as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)
        print(f"Saved {name}: {len(formatted)} items")

    # Save combined cleaned data too
    save_jsonl(balanced, data_dir / "cleaned_data.jsonl")
    print(f"\nAll done! Total dataset: {len(balanced)} items")


if __name__ == "__main__":
    main()
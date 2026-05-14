#!/usr/bin/env python3
"""
Clean augmented data with less aggressive filtering.
Focus on keeping good pairs while removing truly bad ones.
"""

import json
import re
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher


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


def validate_pair(item: dict) -> tuple[bool, str]:
    """Validate a single prompt pair - more lenient."""
    vague = item.get("vague", "")
    optimized = item.get("optimized", "")

    if not vague or not optimized:
        return False, "empty field"

    if len(vague.strip()) < 2:
        return False, "vague too short"

    if len(optimized.strip()) < 20:
        return False, "optimized too short (< 20 chars)"

    if len(optimized) > 3000:
        return False, "optimized too long (> 3000 chars)"

    # Optimized should be at least 1.3x longer than vague
    if len(optimized) < len(vague) * 1.3:
        return False, "optimized not meaningfully improved"

    # Check it's not just a copy
    if optimized.strip().lower() == vague.strip().lower():
        return False, "no optimization"

    return True, "ok"


def score_optimization(item: dict) -> float:
    """Score how good the optimization is (0-1). More lenient than v1."""
    vague = item.get("vague", "")
    optimized = item.get("optimized", "")
    score = 0.0

    # Length ratio (sweet spot: 1.5-6x)
    ratio = len(optimized) / max(len(vague), 1)
    if 2.0 <= ratio <= 6.0:
        score += 0.25
    elif 1.5 <= ratio <= 10.0:
        score += 0.15
    elif 1.3 <= ratio:
        score += 0.05

    # Contains specificity markers
    specificity_patterns = [
        (r'\d+', "number"),              # Numbers
        (r'include[sd]?', "include"),
        (r'format', "format"),
        (r'audience', "audience"),
        (r'[\(\[]\d+', "numbered list"),
        (r'step', "steps"),
        (r'example', "examples"),
        (r'tone\b', "tone"),
        (r'style\b', "style"),
        (r'length|word', "length spec"),
        (r'constrain|require|must', "constraint"),
        (r'focus on', "focus"),
        (r'for\b.*(?:beginner|developer|executive|student|team)', "audience spec"),
    ]
    matches = 0
    for pattern, name in specificity_patterns:
        if re.search(pattern, optimized.lower()):
            matches += 1
    score += min(matches * 0.06, 0.5)

    # Has structure
    if "\n" in optimized or "- " in optimized or "•" in optimized:
        score += 0.05
    if re.search(r'[.:;]', optimized):
        score += 0.05  # Has sentence structure

    # Preserves some original words
    vague_words = set(vague.lower().split()[:5])
    opt_words = set(optimized.lower().split()[:30])
    if vague_words & opt_words:
        score += 0.1

    return min(score, 1.0)


def remove_duplicates(items: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate vague prompts."""
    seen = []
    kept = []
    for item in items:
        vague = item.get("vague", "").lower().strip()
        is_dup = False
        for s in seen:
            if SequenceMatcher(None, vague, s).ratio() > threshold:
                is_dup = True
                break
        if not is_dup:
            seen.append(vague)
            kept.append(item)
    return kept


def main():
    data_dir = Path(__file__).parent.parent / "data"
    raw_dir = data_dir / "raw"

    # Load augmented data (includes seeds + generated + augmented)
    aug_path = raw_dir / "augmented_data.jsonl"
    if not aug_path.exists():
        print(f"Error: {aug_path} not found. Run augment_data.py first.")
        return

    items = load_jsonl(aug_path)
    print(f"Loaded {len(items)} items from augmented data")

    # Validate
    validated = []
    removed_reasons = Counter()
    for item in items:
        valid, reason = validate_pair(item)
        if valid:
            validated.append(item)
        else:
            removed_reasons[reason] += 1

    print(f"After validation: {len(validated)} (removed {len(items) - len(validated)})")
    for reason, count in removed_reasons.most_common():
        print(f"  {reason}: {count}")

    # Score
    for item in validated:
        item["quality_score"] = score_optimization(item)

    # Quality filter - more lenient (min 0.15)
    quality_filtered = [i for i in validated if i["quality_score"] >= 0.15]
    print(f"\nAfter quality filter (min 0.15): {len(quality_filtered)} (removed {len(validated) - len(quality_filtered)})")

    # Remove duplicates
    deduped = remove_duplicates(quality_filtered, threshold=0.85)
    print(f"After deduplication: {len(deduped)} (removed {len(quality_filtered) - len(deduped)})")

    # Category stats
    cat_counts = Counter(i.get("category", "unknown") for i in deduped)
    print(f"\nCategory distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    # Balance: cap over-represented categories at 300
    balanced = []
    cat_limits = {
        "writing": 300, "coding": 250, "analysis": 200, "q_and_a": 200,
        "roleplay": 150, "summarization": 150, "brainstorming": 150,
        "translation": 150, "instruction": 100, "editing": 100, "mixed": 20,
    }

    cat_seen = Counter()
    for item in deduped:
        cat = item.get("category", "unknown")
        limit = cat_limits.get(cat, 200)
        if cat_seen[cat] < limit:
            balanced.append(item)
            cat_seen[cat] += 1

    print(f"\nAfter balancing: {len(balanced)}")
    cat_counts2 = Counter(i.get("category", "unknown") for i in balanced)
    for cat, count in cat_counts2.most_common():
        print(f"  {cat}: {count}")

    # Save cleaned data
    output_path = data_dir / "cleaned_data.jsonl"
    save_jsonl(balanced, output_path)
    print(f"\nSaved {len(balanced)} cleaned items to {output_path}")

    # Score distribution
    scores = [i["quality_score"] for i in balanced]
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\nQuality score: avg={avg:.2f}, min={min(scores):.2f}, max={max(scores):.2f}")

    # Remove quality_score from output (not needed for training)
    for item in balanced:
        item.pop("quality_score", None)
    save_jsonl(balanced, output_path)


if __name__ == "__main__":
    main()
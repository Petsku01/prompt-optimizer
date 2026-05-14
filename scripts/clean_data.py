#!/usr/bin/env python3
"""
Clean and validate prompt optimization dataset.

Filters:
- Remove duplicates (exact and near-duplicate)
- Remove too short/long entries
- Quality score based on optimization dimensions
- Ensure category balance
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
    """Validate a single prompt pair."""
    vague = item.get("vague", "")
    optimized = item.get("optimized", "")

    if not vague or not optimized:
        return False, "empty field"

    if len(vague.strip()) < 3:
        return False, "vague too short"

    if len(optimized.strip()) < 30:
        return False, "optimized too short"

    if len(optimized) > 2000:
        return False, "optimized too long"

    # Optimized should be meaningfully longer than vague
    if len(optimized) < len(vague) * 1.5:
        return False, "optimized not significantly improved"

    # Check optimized isn't just vague repeated
    if optimized.strip().lower() == vague.strip().lower():
        return False, "no optimization"

    # Check optimized has structure (not just run-on sentence)
    has_structure = any([
        "\n" in optimized,
        "-" in optimized,
        "." in optimized[10:] if len(optimized) > 10 else True,
        "," in optimized[10:] if len(optimized) > 10 else True,
    ])

    return True, "ok" if has_structure else "no_structure"


def score_optimization(item: dict) -> float:
    """Score how good the optimization is (0-1). Higher = better."""
    vague = item.get("vague", "")
    optimized = item.get("optimized", "")
    score = 0.0

    # Length ratio (sweet spot: 2-5x)
    ratio = len(optimized) / max(len(vague), 1)
    if 2.0 <= ratio <= 5.0:
        score += 0.2
    elif 1.5 <= ratio <= 8.0:
        score += 0.1

    # Contains specificity markers
    specificity_markers = [
        r'\d+',              # Numbers (word count, steps)
        r'include[sd]?',     # "include"
        r'format',           # Format specification
        r'audience',         # Audience targeting
        r'target',           # Target specification
        r'focus',            # Focus areas
        r'step[s ]',         # Step-by-step
        r'example',          # Request for examples
        r'tone',             # Tone specification
        r'style',            # Style specification
    ]
    matches = sum(1 for m in specificity_markers if re.search(m, optimized.lower()))
    score += min(matches * 0.08, 0.4)

    # Has structure (bullet points, numbered items, headers)
    if "-" in optimized or "•" in optimized:
        score += 0.1
    if re.search(r'\d+\.', optimized):  # Numbered items
        score += 0.1
    if "\n" in optimized:
        score += 0.05

    # Preserves original intent (some overlap with vague)
    vague_words = set(vague.lower().split())
    opt_words = set(optimized.lower().split())
    if vague_words & opt_words:
        score += 0.05

    return min(score, 1.0)


def remove_duplicates(items: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate entries."""
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


def clean_dataset(input_paths: list[Path], output_path: Path, min_score: float = 0.3):
    """Full cleaning pipeline."""
    print("=== Prompt Optimizer Data Cleaning ===\n")

    # Load all data
    all_items = []
    for path in input_paths:
        if path.exists():
            items = load_jsonl(path)
            print(f"Loaded {len(items)} items from {path.name}")
            all_items.extend(items)

    print(f"\nTotal raw items: {len(all_items)}")

    # Validate
    validated = []
    removed_reasons = Counter()
    for item in all_items:
        valid, reason = validate_pair(item)
        if valid:
            validated.append(item)
        else:
            removed_reasons[reason] += 1

    print(f"After validation: {len(validated)} (removed {len(all_items) - len(validated)})")
    for reason, count in removed_reasons.most_common():
        print(f"  {reason}: {count}")

    # Score
    scored = []
    for item in validated:
        item["quality_score"] = score_optimization(item)
        scored.append(item)

    # Filter by quality
    quality_filtered = [i for i in scored if i["quality_score"] >= min_score]
    print(f"\nAfter quality filter (min {min_score}): {len(quality_filtered)} "
          f"(removed {len(scored) - len(quality_filtered)})")

    # Remove duplicates
    deduped = remove_duplicates(quality_filtered)
    print(f"After deduplication: {len(deduped)} (removed {len(quality_filtered) - len(deduped)})")

    # Category balance
    cat_counts = Counter(i.get("category", "unknown") for i in deduped)
    print(f"\nCategory distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    # Save
    save_jsonl(deduped, output_path)
    print(f"\nSaved {len(deduped)} cleaned items to {output_path}")

    # Print score distribution
    scores = [i["quality_score"] for i in deduped]
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\nQuality score: avg={avg:.2f}, min={min(scores):.2f}, max={max(scores):.2f}")

    return deduped


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    raw_dir = data_dir / "raw"

    input_paths = [
        raw_dir / "seed_data.jsonl",
        raw_dir / "generated_data.jsonl",
    ]

    output_path = data_dir / "cleaned_data.jsonl"
    clean_dataset(input_paths, output_path, min_score=0.3)
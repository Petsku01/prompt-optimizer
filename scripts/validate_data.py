#!/usr/bin/env python3
"""
Data Validation Script for Prompt Optimizer
=============================================

Loads all three splits (train.jsonl, val.jsonl, test.jsonl) and validates:
  1. No output leakage between splits
  2. Unique output percentage per category
  3. Output length distribution per category
  4. No remaining quality issues (garbled writing, placeholders, etc.)
  5. Category balance
  6. Minimum per-category sizes

Exits with code 0 if all checks pass, code 1 otherwise.

Usage:
    python scripts/validate_data.py
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter


# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
SPLIT_FILES = ["train.jsonl", "val.jsonl", "test.jsonl"]

MIN_CATEGORY_SIZE = 15
MIN_OUTPUT_UNIQUENESS = 0.50  # Per-category minimum unique output ratio


# ── Checks ────────────────────────────────────────────────────────────────────

def check_output_leakage(splits: dict) -> list:
    """Check that no output appears in multiple splits."""
    issues = []
    split_outputs = {}
    for name, items in splits.items():
        split_outputs[name] = set(normalize(item["output"]) for item in items)

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        overlap = split_outputs[a] & split_outputs[b]
        if overlap:
            issues.append(f"CRITICAL: {len(overlap)} shared outputs between {a} and {b}")
            # Show a few examples
            for ex in list(overlap)[:3]:
                issues.append(f"  Example: {ex[:80]}...")
        else:
            print(f"  ✓ No output leakage between {a} and {b}")

    return issues


def check_unique_outputs(splits: dict) -> list:
    """Check unique output percentage per category."""
    issues = []
    all_items = []
    for items in splits.values():
        all_items.extend(items)

    cat_items = defaultdict(list)
    for item in all_items:
        cat_items[item["category"]].append(item)

    print(f"\n  Category | Items | Unique Outputs | Uniqueness")
    print(f"  {'─'*50}")

    for cat in sorted(cat_items.keys()):
        items = cat_items[cat]
        outputs = [normalize(item["output"]) for item in items]
        unique = len(set(outputs))
        ratio = unique / len(items) if items else 0

        status = "✓" if ratio >= MIN_OUTPUT_UNIQUENESS else "⚠"
        print(f"  {cat:16s} | {len(items):5d} | {unique:5d} | {ratio:.1%}  {status}")

        if ratio < MIN_OUTPUT_UNIQUENESS:
            issues.append(
                f"WARNING: {cat} has only {ratio:.1%} unique outputs "
                f"({unique}/{len(items)}). Threshold: {MIN_OUTPUT_UNIQUENESS:.0%}"
            )

    return issues


def check_output_length(splits: dict) -> list:
    """Report output length distribution per category."""
    issues = []
    all_items = []
    for items in splits.values():
        all_items.extend(items)

    cat_items = defaultdict(list)
    for item in all_items:
        cat_items[item["category"]].append(item)

    print(f"\n  Category | Min | Mean | Max | Std")
    print(f"  {'─'*45}")

    import statistics
    for cat in sorted(cat_items.keys()):
        items = cat_items[cat]
        lengths = [len(item["output"].split()) for item in items]
        print(
            f"  {cat:16s} | {min(lengths):3d} | {statistics.mean(lengths):6.1f} | "
            f"{max(lengths):4d} | {statistics.stdev(lengths):5.1f}"
        )
        # Flag very short outputs (< 10 words)
        short_count = sum(1 for l in lengths if l < 10)
        if short_count > 0:
            issues.append(
                f"WARNING: {cat} has {short_count} outputs shorter than 10 words"
            )

    return issues


def check_quality_issues(splits: dict) -> list:
    """Check for remaining quality issues."""
    issues = []
    all_items = []
    for items in splits.values():
        all_items.extend(items)

    # Check for garbled writing
    garbled = 0
    for item in all_items:
        if item["category"] == "writing":
            output = item["output"]
            has_short = bool(re.search(r'briefly|short response|under \d+ words', output, re.I))
            has_long = bool(re.search(r'\bdetail\b|\bcomprehensively\b|1500\+', output, re.I))
            if has_short and has_long:
                garbled += 1
            if re.search(r'\bWrite a as a\b', output):
                garbled += 1
            if re.match(r'^(As|In|Format as)\s', output) and ' about ' in output:
                garbled += 1
            lengths = re.findall(r'\b(\d{2,4})\s*words?\b', output)
            if len(lengths) >= 2:
                counts = [int(w) for w in lengths]
                if max(counts) > min(counts) * 2:
                    garbled += 1
            if 'paragraphs' in output.lower() and re.search(r'\d+\s*words', output):
                garbled += 1

    if garbled > 0:
        issues.append(f"CRITICAL: {garbled} garbled writing outputs remain")
    else:
        print("  ✓ No garbled writing outputs found")

    # Check for placeholder artifacts
    placeholder = 0
    for item in all_items:
        o = item["output"].lower()
        if re.search(r'\buse\s+none\b', o):
            placeholder += 1
        if '-/-/-' in item["output"]:
            placeholder += 1
        if re.search(r'\bin\s+none\b', o):
            placeholder += 1

    if placeholder > 0:
        issues.append(f"CRITICAL: {placeholder} placeholder artifacts remain")
    else:
        print("  ✓ No placeholder artifacts found")

    # Check for synthetic "please" insertion
    please = 0
    for item in all_items:
        if re.search(r'\b\w+\s+please\s+(me|him|her|us|them|the|a|an|for)\b', item["input"], re.I):
            please += 1
        if re.search(r'\bfor\s+please\b', item["input"], re.I):
            please += 1

    if please > 0:
        issues.append(f"WARNING: {please} broken 'please' insertions remain")
    else:
        print("  ✓ No broken 'please' insertions found")

    # Check for "for for" duplication
    for_for = 0
    for item in all_items:
        if "for for" in item["input"].lower():
            for_for += 1

    if for_for > 0:
        issues.append(f"WARNING: {for_for} 'for for' duplications remain")
    else:
        print("  ✓ No 'for for' duplications found")

    # Check for generic "Option A vs Option B" outputs
    generic = 0
    for item in all_items:
        if re.search(r'Option [AB].*mid-size tech company', item["output"]):
            generic += 1

    if generic > 0:
        issues.append(f"WARNING: {generic} generic 'Option A vs Option B' outputs remain")
    else:
        print("  ✓ No generic template outputs found")

    return issues


def check_category_balance(splits: dict) -> list:
    """Check category distribution across splits is reasonable."""
    issues = []

    cat_counts = {}
    for split_name, items in splits.items():
        cat_counts[split_name] = Counter(item["category"] for item in items)

    all_cats = set()
    for counts in cat_counts.values():
        all_cats.update(counts.keys())

    print(f"\n  Category | Train | Val | Test | Total | Val% | Test%")
    print(f"  {'─'*60}")

    for cat in sorted(all_cats):
        total = sum(cat_counts[s].get(cat, 0) for s in cat_counts)
        if total < MIN_CATEGORY_SIZE:
            issues.append(f"WARNING: {cat} has only {total} items (minimum: {MIN_CATEGORY_SIZE})")

        t = cat_counts["train"].get(cat, 0)
        v = cat_counts["val"].get(cat, 0)
        te = cat_counts["test"].get(cat, 0)
        vp = f"{100*v/total:.0f}%" if total > 0 else "N/A"
        tp = f"{100*te/total:.0f}%" if total > 0 else "N/A"
        print(f"  {cat:16s} | {t:5d} | {v:3d} | {te:4d} | {total:5d} | {vp:>4s} | {tp:>5s}")

    # Check that all splits have all categories
    for split_name in cat_counts:
        missing = all_cats - set(cat_counts[split_name].keys())
        if missing:
            issues.append(f"WARNING: {split_name} split is missing categories: {missing}")
        for cat in cat_counts[split_name]:
            if cat_counts[split_name][cat] == 0:
                issues.append(f"WARNING: {split_name} split has 0 items for {cat}")

    return issues


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PROMPT OPTIMIZER DATA VALIDATION")
    print("=" * 60)

    all_issues = []

    # Load splits
    splits = {}
    for fname in SPLIT_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  ERROR: {path} not found!")
            all_issues.append(f"CRITICAL: Missing file {path}")
            continue

        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                # Convert format if needed
                if "vague" in item:
                    converted = {
                        "instruction": "Transform the following vague prompt into a clear, specific, and effective prompt:",
                        "input": item["vague"],
                        "output": item["optimized"],
                        "category": item["category"],
                        "system": "You are a prompt optimization assistant.",
                    }
                else:
                    converted = item
                items.append(converted)

        splits[fname.replace(".jsonl", "")] = items
        print(f"  Loaded {path}: {len(items)} items")

    if len(splits) < 3:
        print("\n  CRITICAL: Not all split files found. Cannot validate.")
        sys.exit(1)

    # Run checks
    print("\n[1/5] Checking output leakage between splits...")
    all_issues.extend(check_output_leakage(splits))

    print("\n[2/5] Checking unique output percentages...")
    all_issues.extend(check_unique_outputs(splits))

    print("\n[3/5] Checking output length distributions...")
    all_issues.extend(check_output_length(splits))

    print("\n[4/5] Checking for quality issues...")
    all_issues.extend(check_quality_issues(splits))

    print("\n[5/5] Checking category balance...")
    all_issues.extend(check_category_balance(splits))

    # Summary
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)

    if not all_issues:
        print("  ✅ All checks passed!")
        sys.exit(0)
    else:
        critical = [i for i in all_issues if i.startswith("CRITICAL")]
        warnings = [i for i in all_issues if i.startswith("WARNING")]

        if critical:
            print(f"\n  ❌ {len(critical)} CRITICAL issues found:")
            for issue in critical:
                print(f"    {issue}")
        if warnings:
            print(f"\n  ⚠️  {len(warnings)} warnings found:")
            for issue in warnings:
                print(f"    {issue}")

        if critical:
            print("\n  Exiting with code 1 due to critical issues.")
            sys.exit(1)
        else:
            print("\n  No critical issues. Warnings should be reviewed but are not blockers.")
            sys.exit(0)


if __name__ == "__main__":
    main()
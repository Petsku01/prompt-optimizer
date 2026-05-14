#!/usr/bin/env python3
"""
Evaluate the prompt optimizer model.

Compares baseline (base model) vs finetuned model on test set.
Measures: exact match, length ratio, specificity improvement, category coverage.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def count_specificity_markers(text: str) -> int:
    """Count how many optimization dimensions a text addresses."""
    patterns = [
        (r'\d+\s*(words?|steps?|items?|examples?|points?|paragraphs?)', "length"),
        (r'(format|structure|organize|list|table|bullets?|numbered)', "format"),
        (r'(audience|for\s+(a|an|the)\s+\w+|beginner|expert|developer|student|executive)', "audience"),
        (r'(tone|style|professional|conversational|formal|casual|academic|friendly)', "tone"),
        (r'(include|cover|focus|address|mention|provide|ensure)', "specificity"),
        (r'(constraint|require|must|should|avoid|don\'t|never|always)', "constraint"),
        (r'(example|instance|such as|e\.g\.|case study)', "examples"),
    ]
    count = 0
    for pattern, name in patterns:
        if re.search(pattern, text.lower()):
            count += 1
    return count


def evaluate_pair(vague: str, optimized: str, reference: str = None) -> dict:
    """Evaluate a single optimization pair."""
    result = {
        "vague_length": len(vague),
        "optimized_length": len(optimized),
        "length_ratio": len(optimized) / max(len(vague), 1),
        "specificity_markers": count_specificity_markers(optimized),
        "has_structure": bool(re.search(r'[\n\-•]', optimized)),
        "has_numbers": bool(re.search(r'\d+', optimized)),
        "has_format_spec": bool(re.search(r'(format|structure|list|table|step)', optimized.lower())),
        "has_audience": bool(re.search(r'(audience|for\s+(a|an|the)\s+\w+|beginner|expert)', optimized.lower())),
        "has_tone": bool(re.search(r'(tone|professional|conversational|formal|casual)', optimized.lower())),
    }

    if reference:
        # Compare with reference
        result["reference_length"] = len(reference)
        result["reference_markers"] = count_specificity_markers(reference)
        # Word overlap
        opt_words = set(optimized.lower().split())
        ref_words = set(reference.lower().split())
        if opt_words and ref_words:
            result["word_overlap"] = len(opt_words & ref_words) / len(opt_words | ref_words)

    return result


def evaluate_dataset(test_data: list[dict]) -> dict:
    """Evaluate entire test dataset."""
    results = []
    category_results = defaultdict(list)

    for item in test_data:
        vague = item.get("input", item.get("vague", ""))
        optimized = item.get("output", item.get("optimized", ""))
        category = item.get("category", "unknown")

        if not vague or not optimized:
            continue

        eval_result = evaluate_pair(vague, optimized)
        results.append(eval_result)
        category_results[category].append(eval_result)

    # Aggregate stats
    if not results:
        return {}

    avg_markers = sum(r["specificity_markers"] for r in results) / len(results)
    avg_ratio = sum(r["length_ratio"] for r in results) / len(results)
    pct_structured = sum(1 for r in results if r["has_structure"]) / len(results) * 100
    pct_has_format = sum(1 for r in results if r["has_format_spec"]) / len(results) * 100
    pct_has_audience = sum(1 for r in results if r["has_audience"]) / len(results) * 100
    pct_has_tone = sum(1 for r in results if r["has_tone"]) / len(results) * 100

    report = {
        "total_examples": len(results),
        "avg_specificity_markers": round(avg_markers, 2),
        "avg_length_ratio": round(avg_ratio, 2),
        "pct_structured": round(pct_structured, 1),
        "pct_has_format": round(pct_has_format, 1),
        "pct_has_audience": round(pct_has_audience, 1),
        "pct_has_tone": round(pct_has_tone, 1),
        "categories": {},
    }

    for cat, cat_results in category_results.items():
        report["categories"][cat] = {
            "count": len(cat_results),
            "avg_markers": round(sum(r["specificity_markers"] for r in cat_results) / len(cat_results), 2),
            "avg_ratio": round(sum(r["length_ratio"] for r in cat_results) / len(cat_results), 2),
        }

    return report


def print_report(report: dict, title: str = "Evaluation Report"):
    """Print a formatted report."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Total examples:   {report['total_examples']}")
    print(f"  Avg specificity:   {report['avg_specificity_markers']}/7 dimensions")
    print(f"  Avg length ratio:  {report['avg_length_ratio']}x (optimized/vague)")
    print(f"  Has structure:     {report['pct_structured']}%")
    print(f"  Has format spec:   {report['pct_has_format']}%")
    print(f"  Has audience:      {report['pct_has_audience']}%")
    print(f"  Has tone:          {report['pct_has_tone']}%")
    print(f"\n  Per-category breakdown:")
    for cat, stats in report.get("categories", {}).items():
        print(f"    {cat:15s}: {stats['count']:3d} examples, "
              f"markers={stats['avg_markers']:.1f}, ratio={stats['avg_ratio']:.1f}x")


def interactive_test(model_path: str = None):
    """Interactive testing of the model."""
    print("\n" + "="*60)
    print("  Interactive Prompt Optimizer Test")
    print("  Type 'quit' to exit, 'stats' for dataset stats")
    print("="*60 + "\n")

    # Try to load model if path provided
    model = None
    tokenizer = None
    if model_path and Path(model_path).exists():
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(model_path)
            FastLanguageModel.for_inference(model)
            print(f"Model loaded from {model_path}")
        except ImportError:
            print("Unsloth not available. Running in dataset evaluation mode only.")

    test_path = Path(__file__).parent.parent / "data" / "test.jsonl"
    test_data = load_jsonl(test_path) if test_path.exists() else []

    # Evaluate test dataset
    if test_data:
        report = evaluate_dataset(test_data)
        print_report(report, "Test Dataset Quality Report")

    while True:
        try:
            prompt = input("\nEnter a vague prompt to optimize: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() == "quit":
            break
        elif prompt.lower() == "stats":
            if test_data:
                report = evaluate_dataset(test_data)
                print_report(report)
            continue
        elif not prompt:
            continue

        if model and tokenizer:
            SYSTEM_PROMPT = "You are a prompt engineering expert that transforms vague, underspecified prompts into clear, well-structured, and effective prompts. You preserve the user's original intent while adding relevant specificity, format constraints, audience targeting, and contextual details."
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Optimize the following prompt to be clear, specific, and effective. Preserve the original intent while adding structure, context, and constraints where appropriate.\n\nPrompt to optimize: {prompt}"},
            ]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
            outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True)
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            print(f"\nOptimized: {response.strip()}")
            eval_result = evaluate_pair(prompt, response.strip())
            print(f"Specificity: {eval_result['specificity_markers']}/7 | Length ratio: {eval_result['length_ratio']:.1f}x")
        else:
            print("No model loaded. Showing dataset example instead:")
            if test_data:
                import random
                example = random.choice(test_data)
                print(f"\nVague: {example.get('input', example.get('vague', ''))}")
                print(f"Optimized: {example.get('output', example.get('optimized', ''))}")


def main():
    data_dir = Path(__file__).parent.parent / "data"

    # Evaluate all splits
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            data = load_jsonl(path)
            report = evaluate_dataset(data)
            print_report(report, f"{split.upper()} Dataset Report")
        else:
            print(f"{split}.jsonl not found at {path}")

    # Interactive test (without model)
    if "--interactive" in sys.argv:
        model_path = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
        interactive_test(model_path)


if __name__ == "__main__":
    main()
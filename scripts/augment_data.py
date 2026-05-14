#!/usr/bin/env python3
"""
Augment dataset by creating variations of existing pairs.
Adds: paraphrases, topic swaps, and dimension combinations.
"""

import json
import random
from pathlib import Path

random.seed(123)


def augment_vague(vague: str) -> list[str]:
    """Create variations of a vague prompt."""
    variations = [vague]

    # Paraphrase patterns
    replacements = {
        "write about": ["tell me about", "explain", "describe", "give me info on", "I want to know about"],
        "write me": ["create for me", "generate", "produce", "compose", "draft"],
        "fix my": ["debug this", "help me fix", "what's wrong with my", "troubleshoot my"],
        "how to": ["guide me through", "walk me through", "teach me", "steps for", "instructions for"],
        "give me ideas": ["suggest", "brainstorm", "come up with", "propose", "I need ideas for"],
        "summarize this": ["give me the key points of", "what's the gist of", "condense", "provide a brief overview of"],
        "improve this": ["make better", "enhance", "polish", "refine", "revise"],
        "analyze": ["examine", "break down", "evaluate", "assess", "study"],
        "compare": ["contrast", "weigh the pros and cons of", "evaluate the differences between"],
        "translate this": ["convert to", "render in", "provide a version in"],
        "act as": ["pretend to be", "roleplay as", "simulate being", "take the role of"],
    }

    for old, new_options in replacements.items():
        if old in vague.lower():
            for new in new_options:
                varied = vague.lower().replace(old, new, 1)
                # Capitalize first letter
                varied = varied[0].upper() + varied[1:] if varied else varied
                if varied != vague:
                    variations.append(varied)

    # Add/remove punctuation variations
    if vague.endswith("."):
        variations.append(vague[:-1])
    elif not vague.endswith("?") and vague.split()[0].lower() in ["what", "how", "why", "when", "where"]:
        variations.append(vague + "?")
    else:
        variations.append(vague + ".")

    # Add "please" variation
    if "please" not in vague.lower():
        words = vague.split()
        # Insert "please" after first verb-like word
        verbs = ["write", "create", "make", "help", "fix", "analyze", "explain", "summarize", "translate", "improve", "generate", "suggest", "teach", "guide", "compare", "evaluate", "compose", "draft", "build", "implement"]
        for i, w in enumerate(words):
            if w.lower() in verbs:
                words.insert(i + 1, "please")
                variations.append(" ".join(words))
                break

    return list(set(variations))[:5]  # Max 5 variations


def augment_optimized(optimized: str) -> list[str]:
    """Create variations of optimization by adding/removing dimensions."""
    variations = [optimized]

    # Add constraint variations
    extras = [
        " Be concise and specific.",
        " Use clear, direct language.",
        " Prioritize accuracy over comprehensiveness.",
        " Include relevant examples where helpful.",
        " Focus on practical, actionable advice.",
        " Maintain a neutral, objective tone.",
        " Avoid unnecessary jargon.",
        " Structure your response for easy scanning.",
    ]

    # Add 1-2 random extras
    selected = random.sample(extras, min(2, len(extras)))
    for extra in selected:
        if extra.strip() not in optimized:
            variations.append(optimized + extra)

    return list(set(variations))[:3]


def create_cross_category_pairs() -> list[dict]:
    """Create mixed-category vague->optimized pairs."""
    pairs = []

    # Writing + Coding = "explain this code in writing"
    coding_writing = [
        {"vague": "document this code", "optimized": "Write clear, comprehensive documentation for the following code. Include: purpose and overview, function-by-function explanation with parameter descriptions, usage examples, edge cases, and return values. Format as a well-structured markdown document suitable for onboarding new developers."},
        {"vague": "comment this function", "optimized": "Add docstrings and inline comments to the following function. For each docstring include: purpose, parameters (with types and descriptions), return value, raised exceptions, and one usage example. Follow Google-style docstring format. Keep comments concise but informative."},
        {"vague": "explain this error", "optimized": "Analyze the following error message and stack trace. Provide: (1) a plain-language explanation of what went wrong, (2) the most likely root cause, (3) step-by-step instructions to fix it, and (4) how to prevent similar errors. Assume the reader is a junior developer."},
    ]

    # Analysis + Q&A = "analyze and explain"
    analysis_qa = [
        {"vague": "why is this slow", "optimized": "Analyze the performance bottleneck in the following code or system. Identify: (1) the specific slow operation, (2) time complexity analysis, (3) memory usage concerns, (4) 3 concrete optimization strategies with expected impact. Present findings as a structured performance report with before/after estimates."},
        {"vague": "is this secure", "optimized": "Conduct a security review of the following code/architecture. Check for: SQL injection, XSS, CSRF, authentication bypass, insecure data storage, and OWASP Top 10 vulnerabilities. For each finding provide: severity (Critical/High/Medium/Low), description, proof of concept, and specific fix. Format as a security audit report."},
    ]

    # Summarization + Editing = "improve and condense"
    summary_editing = [
        {"vague": "clean this up", "optimized": "Review and improve the following text. Apply these changes: (1) remove redundancy and filler, (2) fix grammar and awkward phrasing, (3) improve clarity and flow, (4) add brief summaries where sections are long. Target length: 60-70% of original. Preserve all factual content and original intent."},
        {"vague": "make this shorter and better", "optimized": "Condense the following text to approximately half its current length while improving clarity. Specifically: eliminate passive voice, remove redundant adjectives, merge overlapping points, and replace vague statements with specific ones. Maintain the original tone and all key information. Add a one-sentence summary at the top."},
    ]

    pairs.extend(coding_writing)
    pairs.extend(analysis_qa)
    pairs.extend(summary_editing)

    for p in pairs:
        p["category"] = "mixed"

    return pairs


def main():
    data_dir = Path(__file__).parent.parent / "data" / "raw"

    # Load existing data
    existing = []
    for fname in ["seed_data.jsonl", "generated_data.jsonl"]:
        fpath = data_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        existing.append(json.loads(line))

    print(f"Loaded {len(existing)} existing pairs")

    # Augment by creating variations of vague prompts
    augmented = []
    seen_vague = set(p["vague"].lower().strip() for p in existing)

    for item in existing:
        vague_vars = augment_vague(item["vague"])
        for v in vague_vars:
            key = v.lower().strip()
            if key not in seen_vague:
                seen_vague.add(key)
                # Keep the same optimized output (the vague change is the variation)
                augmented.append({
                    "vague": v,
                    "optimized": item["optimized"],
                    "category": item.get("category", "unknown"),
                })

        # Add optimization variations
        opt_vars = augment_optimized(item["optimized"])
        for o in opt_vars:
            if o != item["optimized"]:
                augment_key = f"{item['vague']}|||{o}"
                # Only add if we haven't seen this exact pair
                augmented.append({
                    "vague": item["vague"],
                    "optimized": o,
                    "category": item.get("category", "unknown"),
                })

    # Add cross-category pairs
    cross = create_cross_category_pairs()
    augmented.extend(cross)

    # Merge all
    all_pairs = existing + augmented
    seen_keys = set()
    unique_pairs = []
    for p in all_pairs:
        key = f"{p['vague']}|||{p['optimized']}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique_pairs.append(p)

    print(f"After augmentation: {len(augmented)} new pairs")
    print(f"Total unique pairs: {len(unique_pairs)}")

    # Category stats
    from collections import Counter
    cat_counts = Counter(p.get("category", "unknown") for p in unique_pairs)
    print("\nCategory distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    # Save
    output_path = data_dir / "augmented_data.jsonl"
    with open(output_path, "w") as f:
        for p in unique_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
# Data Quality Review: Prompt Optimizer Dataset

**Reviewer:** Senior ML Engineer  
**Date:** 2026-05-14  
**Dataset:** Cleaned prompt optimizer training data (969 items)  
**Purpose:** QLoRA finetuning of Qwen2.5-3B-Instruct  

---

## Executive Summary

The cleaned dataset has serious quality problems that will materially harm model performance. Despite the cleaning pipeline removing near-duplicates and filtering short writing outputs, the remaining data suffers from: (1) massive output duplication where different inputs map to identical or near-identical outputs, (2) severe template boilerplate across most categories, (3) latent data leakage between splits via shared outputs, (4) garbled writing-category outputs, and (5) placeholder artifacts in brainstorming items. I recommend **not training on this dataset in its current form**.

---

## 1. Remaining Quality Issues in the Cleaned Data

### 1.1 CRITICAL: Output Duplication Crisis

**Only 30.7% of outputs are unique across the full dataset.** The Jaccard deduplication on inputs missed the real problem: many different vague inputs map to the same optimized output. This is the single most damaging issue.

| Category | Items | Unique Outputs | Uniqueness |
|---|---:|---:|---:|
| q_and_a | 112 | 12 | 10.7% |
| roleplay | 80 | 10 | 12.5% |
| coding | 188 | 32 | 17.0% |
| translation | 42 | 9 | 21.4% |
| summarization | 24 | 9 | 37.5% |
| brainstorming | 127 | 49 | 38.6% |
| editing | 94 | 47 | 50.0% |
| instruction | 90 | 43 | 47.8% |
| analysis | 175 | 53 | 30.3% |
| writing | 37 | 33 | 89.2% |

**Examples of the problem:**
- 11 different "explain DNS" inputs all produce the **exact same output**: "Explain how DNS works step by step, from typing a URL to the page loading. Include: recursive and iterative queries, caching, TTL, and common DNS records..."
- 13 coding items share the output "Implement a caching layer with LRU eviction policy and Redis backend..." despite having different inputs like "refactor my caching layer" vs "fix my caching layer"
- 51 analysis items (29% of that category) start with the identical template "Compare Option A vs Option B for a mid-size tech company..."

**Impact:** The model will learn to produce the same output regardless of input nuance. For a prompt optimizer, this is fatal — it should produce *different* optimized prompts for different vague inputs.

### 1.2 CRITICAL: Data Leakage Between Splits

While input-level deduplication ensures no shared inputs between train/val/test, the output-level leakage is severe:

- **Val items sharing an output with train:** 84/93 (90.3%)
- **Test items sharing an output with train:** 90/104 (86.5%)

This means evaluation metrics will be misleadingly high. Val/test items with outputs that appear in train don't test generalization — they test memorization.

### 1.3 SEVERE: Template Boilerplate in Outputs

The outputs are overwhelmingly formulaic. Each category uses 2-4 recycled templates with minor topic substitution:

**Instruction category (90 items, only 5 actual templates):**
- "Write a step-by-step guide for {topic}. Target audience: {audience}. Include: prerequisites, each step..."
- "Create a structured learning guide for {topic}. Structure: (1) prerequisites and assumed knowledge..."
- "Create an interactive tutorial for {topic}. Each section should: explain the concept..."
- "Write a practical guide for {topic} covering: setup and prerequisites..."

**Brainstorming category (3 templates shared across all 127 items):**
- "Generate {N} creative ideas for {topic}. For each idea, provide: (1) a concise name, (2) the problem it solves..."
- "Brainstorm {N} approaches to {topic}. Categorize each as: Quick Win/Medium Effort/Long Term..."
- "Help me explore {topic} from multiple angles. Provide: (1) 5 conventional approaches..."

**Analysis category (2 dominant templates):**
- "Compare Option A vs Option B for a mid-size tech company..." (51 items!)
- "Conduct a thorough evaluation of {topic} covering: {aspects}. Prioritize findings by impact level..."

**Q&A category (1 template for nearly all 112 items):**
- "Explain {topic} for {audience}. Cover: {list}. Include..."

**Impact:** The model will learn these rigid templates rather than learning *how* to optimize prompts. It will produce formulaic, interchangeable outputs regardless of the input's specific needs.

### 1.4 SEVERE: Writing Category Outputs Are Garbled

The writing category was cleaned with a 120-character minimum, but the outputs that survived are **incoherent mashups** of conflicting instructions:

| Input | Output |
|---|---|
| "craft a story about artificial intelligence." | "Write a briefly (150-200 words) story about artificial intelligence. in an authoritative tone for developers in detail (1500+ words)" |
| "compose please a guide about cultural diversity" | "Write a as a short response (under 100 words) guide about cultural diversity. in an authoritative tone for industry professionals" |
| "construct a guide about renewable energy." | "Write a detail (1500+ words) guide about renewable energy. Focus on future outlook. in a humorous tone where appropriate" |
| "write me article about street food." | "Write a briefly (150-200 words) article about street food. Focus on practical applications. in 500 words in a formal tone" |

These outputs contain **self-contradictory length instructions** ("briefly 150-200 words" + "in detail 1500+ words"), **incoherent phrasing** ("Write a as a short response"), and **grammatically broken structures**. The augmentation/dgeneration process clearly failed for this category.

Only 2 of the 37 writing items are well-formed (the blog post and story about Arctic station — which are seed data).

### 1.5 MODERATE: Placeholder Artifacts

**"Use none" in instruction category:** Several instruction outputs contain "Include code examples in none" or "Use none for all examples" — these are clearly template-filling failures where no programming language was inserted.

**"-/-/-" in brainstorming:** Multiple brainstorming items contain "estimated complexity to implement (-/-/-)" instead of actual Low/Medium/High ratings.

### 1.6 MODERATE: Grammatically Broken Inputs

124 items (12.8%) contain awkwardly placed "please" artifacts from augmentation:
- "teach please me configuring a firewall"
- "guide please for writing unit tests"
- "compose please a guide about cultural diversity"

And 8 items contain "for for" duplication:
- "I need ideas for for community events"
- "I need ideas for for cost reduction"

These are not "vague user prompts" — they are obviously synthetic artifacts.

---

## 2. Cleaning Script Review (`scripts/clean_and_split.py`)

### 2.1 What It Does Well

- **Stratified splitting** ensures category proportions are maintained across train/val/test.
- **Seed-based randomness** (SEED=42) makes the process reproducible.
- **Mixed category removal** is reasonable given the small count (7 items).
- **Short writing output filtering** (MIN_OUTPUT_LEN_WRITING=120) is a good idea, though the threshold is too low.

### 2.2 Critical Flaws

**Flaw 1: Deduplication on inputs, not outputs.**

The Jaccard deduplication only compares vague prompts (inputs), which means it preserves items with vastly different inputs but identical outputs. This is the root cause of the output duplication crisis. The deduplication should operate on output similarity, or at minimum on the input-output pair.

**Flaw 2: Jaccard on word sets is too coarse.**

`word_set(s)` splits on whitespace and lowercases. This means "how to docker" and "how to Docker" are near-identical (good), but it also means "guide for writing unit tests" and "teach me writing unit tests" have a Jaccard of only 0.43 despite semantically requesting the same thing. Meanwhile, "guide for writing unit tests" and "guide for implementing CI/CD" have a Jaccard of 0.71 despite needing completely different outputs.

**Flaw 3: Quality filter is too weak.**

The only quality filter is `len(item["optimized"]) < 120` for writing. This misses:
- Self-contradictory instructions (writing)
- Placeholder artifacts ("Use none", "-/-/-")
- Grammatically broken outputs
- Template-filling failures
- All problems in non-writing categories

**Flaw 4: No output-diversity enforcement.**

The script has no mechanism to detect or deduplicate near-identical outputs. Even a simple check like `if output already seen: skip` would have caught the exact duplicates.

### 2.3 Script Logic Issues

```python
# Line 40: Prefers items with longer outputs during dedup
data_sorted = sorted(data, key=lambda x: len(x["optimized"]), reverse=True)
```

This is reasonable for keeping the most detailed version, but it means the script keeps the *longest* output, not the *best* one. Given the writing category issues, longer outputs are often worse (more garbled).

```python
# Lines 80-84: Split logic can produce 0-item val splits
n_val = max(1, int(n * VAL_RATIO))
if n - n_train - n_val < 1:
    n_val = max(0, n - n_train - 1)
```

For very small categories (summarization has 24 items), this can create edge cases. For 24 items: n_train=19, n_val=2, n_test=3. But the val set for summarization only has 2 items, making it unusable for category-stratified evaluation.

---

## 3. Template-Like Patterns by Category

### Categories ranked by output diversity (worst to best):

| Rank | Category | Unique Templates | Description |
|---|---|---:|---|
| 1 | q_and_a | ~1 | "Explain X for Y. Cover: A, B, C. Include..." |
| 2 | instruction | ~5 | "Write a step-by-step guide..." / "Create a structured learning guide..." |
| 3 | roleplay | ~8 | "Act as a X with Y years of experience..." |
| 4 | brainstorming | ~3 | "Generate N creative ideas..." / "Brainstorm N approaches..." |
| 5 | coding | ~10 | Various but very similar structures |
| 6 | translation | ~7 | "Translate the following X to Y. Use Z..." |
| 7 | summarization | ~7 | "Summarize X in Y bullet points..." |
| 8 | editing | ~6 | "Improve/Enhance/Rewrite this X by..." |
| 9 | analysis | ~5 | "Compare Option A vs Option B..." / "Conduct a thorough evaluation..." |
| 10 | writing | ~varied | Diverse but incoherent |

The model will learn category-specific templates rather than general prompt optimization principles. When evaluated on a new prompt that doesn't fit these templates, it will fail.

---

## 4. Output Quality Assessment: Formulaic, Not Diverse

**The core problem:** The data was generated using a process that creates topic substitutions within rigid templates, not genuine prompt optimization. Evidence:

1. **Mechanical topic filling:** Every Q&A output uses "Explain {topic} for {audience}. Cover: {list}." The topic and audience swap but the structure is identical.

2. **Template count vs item count:** The entire dataset of 969 items appears to be generated from approximately **40-50 unique output templates**. This means the model will see each template ~20 times on average, reinforcing boilerplate.

3. **Contradictory diversity in augmentation:** The `augment_data.py` script creates input variations (paraphrasing "how to" -> "guide me through" -> "teach me") but keeps the **same output**, teaching the model that completely different inputs should produce identical outputs.

4. **Good examples exist but are rare:** The seed data items (e.g., "act as a career counselor" -> the detailed counseling prompt, "how to set up docker" -> the specific Ubuntu 22.04 guide) show what high-quality optimization looks like. But these are drowned out by hundreds of template-filled items.

---

## 5. Category-Specific Problems

### coding (188 items, 17% unique outputs)
- **Template reuse:** Only ~10 distinct output structures for 188 items
- **Same task, different verb problem:** "fix my caching layer", "refactor my caching layer", "debug my caching layer" all produce the same output. The model learns that optimization is verb-insensitive, which is wrong — "fix" should produce a debugging-focused prompt, "refactor" should produce a restructuring prompt.

### writing (37 items, 89% unique outputs)
- **Outputs are incoherent** as detailed in 1.4. Only the seed data items are usable.
- **Even after the 120-char filter**, surviving items like "Write a as a short response (under 100 words) guide about cultural diversity" are nonsensical.

### analysis (175 items, 30% unique outputs)
- **51 items (29%)** produce "Compare Option A vs Option B for a mid-size tech company..." regardless of whether the input is about energy consumption, team productivity, or investment opportunities. This is pure template filling.
- Inputs like "evaluate technology adoption rates" produce outputs about comparing "Option A vs Option B" — the topic is lost entirely.

### instruction (90 items, 48% unique outputs)
- **"Use none" artifact:** "Include code examples in none" appears in multiple tutorial outputs.
- Only ~5 actual templates for 90 items.

### brainstorming (127 items, 39% unique outputs)
- **"-/-/-" artifacts:** "estimated complexity to implement (-/-/-)" instead of Low/Medium/High
- Only 3 templates for all 127 items

### translation (42 items, 21% unique outputs)
- Relatively small category with only 5-6 document types (academic paper, website UI, social media, technical docs, product description) × 5-6 languages, producing highly repetitive outputs.

### roleplay (80 items, 12.5% unique outputs)
- Only ~8 unique roleplay outputs. "Act as a financial advisor" → identical output whether input is "act as a financial advisor", "be a financial advisor", "pretend to be a financial advisor", or "roleplay as a financial advisor"

### summarization (24 items, 37.5% unique outputs)
- Far too few items for the model to learn this category
- Only 2 items in val set — insufficient for meaningful evaluation

---

## 6. Dataset Size Assessment for QLoRA Finetuning

### Is 969 items sufficient?

**For a 3B model with QLoRA:** Technically, yes — QLoRA updates only 0.1-1% of parameters, so 770 training items could be enough to adapt behavior. However:

**Effective unique training signal:** Only ~297 unique outputs across 772 training items. With heavy template reuse, the effective diversity is closer to **40-50 template patterns**. This means QLoRA will:
- Overfit to template structures
- Fail to generalize to novel prompt types
- Not learn the *skill* of prompt optimization — just the *patterns* of specific templates

**Recommended minimum for production quality:** 2,000-5,000 items with genuinely diverse outputs (each output structurally unique, not just topic-substituted). For this task, quality matters far more than quantity — 500 diverse, high-quality pairs would outperform 5,000 template-filled items.

---

## 7. Recommendations

### MUST FIX (Before Training)

**1. Deduplicate on outputs, not just inputs.** Add output deduplication:
```python
def deduplicate_outputs(data: list[dict]) -> list[dict]:
    seen_outputs = set()
    unique = []
    for item in data:
        output_key = item["optimized"].strip().lower()
        if output_key not in seen_outputs:
            seen_outputs.add(output_key)
            unique.append(item)
    return unique
```
This alone would reduce the dataset from 969 to ~297 items — but those 297 would each provide unique learning signal.

**2. Remove all garbled writing outputs.** Any writing output containing contradictory length instructions or broken grammar ("Write a as a short response", "briefly (150-200 words)" + "in detail (1500+ words)") should be removed. This removes ~35 of 37 writing items.

**3. Remove placeholder artifacts.** Strip items containing "Use none", "-/-/-", or empty template fields.

**4. Fix data leakage between splits.** After deduplication, ensure no output appears in more than one split. Group identical-output items together before splitting.

### SHOULD FIX (Significant Quality Improvement)

**5. Regenerate or heavily curate the data.** The current dataset was generated by LLM with topic-substitution templates. Replace with genuinely diverse outputs:
- Each input should produce a *structurally different* optimized prompt
- An "explain DNS" prompt could become a comparison table, a step-by-step walkthrough, an FAQ, or an analogy-based explanation — not always "Explain X for Y. Cover: A, B, C."
- Vary output length dramatically (50-500 characters)
- Include outputs with markdown, numbered lists, constraints in different positions

**6. Remove or redo brainstorming and analysis categories.** These have the worst template-to-item ratios. "Generate N creative ideas for X" should not be the answer to every brainstorming prompt.

**7. Add quality validation for outputs.** Check for:
- Self-contradictory constraints
- Grammatical errors in outputs
- Placeholder values that were never filled
- Outputs that don't reference the input topic (e.g., "Option A vs Option B" pattern)

### NICE TO HAVE (Marginal Improvement)

**8. Increase summarization and translation data.** Current counts (24 and 42) are too low. Either generate more diverse items or merge summarization into editing and translation into a broader "transformation" category.

**9. Remove or label synthetic "please" insertions** and "for for" duplications from inputs — these don't represent real user patterns.

**10. Add a held-out evaluation set with hand-crafted items** that intentionally challenge the model's ability to handle novel input types not in training data.

---

## Summary of Findings

| Issue | Severity | Impact |
|---|---|---|
| Output duplication (only 30.7% unique) | CRITICAL | Model learns templates, not skill |
| Data leakage (90% of val/test outputs in train) | CRITICAL | Evaluation metrics will be inflated |
| Writing outputs are garbled | CRITICAL | Model may learn incoherent patterns |
| Template boilerplate (4-5 templates per category) | SEVERE | Formulaic, non-diverse model output |
| Placeholder artifacts ("none", "-/-/-") | MODERATE | Model may output literal "none" |
| Synthetic input artifacts ("teach please me") | MODERATE | Model learns unnatural input patterns |
| Small categories (summarization 24, translation 42) | MODERATE | Poor performance on rare categories |
| No output-level deduplication | CRITICAL | Root cause of duplication issues |

**Bottom line:** The dataset is not ready for training. The cleaning script addressed easy surface problems (near-duplicate inputs, short writing, mixed category) but missed the structural issue: the data generation process created template-filled outputs with negligible diversity. A QLoRA model trained on this data will produce formulaic, template-locked outputs that fail on any prompt type not in the training set.
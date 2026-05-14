# Prompt Optimizer - Fine-Tuning Project

## Overview

Fine-tune a small LLM (Qwen2.5-3B-Instruct) to automatically transform vague, poorly-structured prompts into clear, effective, well-structured prompts.

**Goal:** Model that takes a rough prompt as input and outputs an optimized version with:
- Clear role/status assignment
- Specific task description
- Output format constraints
- Tone and audience specification
- Relevant context inclusion

## Architecture

- **Base model:** Qwen2.5-3B-Instruct
- **Method:** QLoRA (4-bit quantization, LoRA adapters)
- **Framework:** Unsloth (2-5x faster training)
- **Platform:** Google Colab (T4 GPU, 16GB VRAM)
- **Target dataset:** 2000+ prompt pairs (vague -> optimized)

## Data Format

Each training example is an instruction-tuning pair:

```json
{
  "instruction": "Optimize the following prompt to be clear, specific, and effective. Preserve the original intent while adding structure, context, and constraints.",
  "input": "write me an essay about dogs",
  "output": "Write a 500-word informative essay about working dog breeds, covering their historical roles, modern applications, and training requirements. Use a professional tone and support your claims with examples."
}
```

## Prompt Categories

| Category | Description | Target Count |
|----------|-------------|--------------|
| writing | Creative and formal writing | 300 |
| coding | Programming and debugging | 300 |
| analysis | Data analysis, reasoning, comparison | 250 |
| translation | Language translation tasks | 200 |
| q&a | Factual questions, explanations | 250 |
| roleplay | Role-based scenarios | 150 |
| summarization | Summarizing text, meetings | 200 |
| brainstorming | Ideas generation, planning | 150 |
| instruction | Step-by-step guides, tutorials | 100 |
| editing | Rewriting, improving text | 100 |
| **total** | | **2000** |

## Optimization Dimensions

A good optimized prompt should address some (not necessarily all) of:

1. **Role/Persona** - "You are an expert in..."
2. **Task specificity** - Clear, unambiguous action verb
3. **Output format** - JSON, table, bullet points, essay
4. **Constraints** - Word count, tone, audience, style
5. **Context** - Background info, domain, scenario
6. **Examples** - Few-shot demonstrations (implicit in structure)
7. **Edge cases** - What to avoid, what to include

## Quality Criteria for Training Data

### Vague prompts (input) should be:
- Short (1-15 words typically)
- Ambiguous or underspecified
- Representative of real user behavior
- Diverse in domain and language (mostly English, some Finnish)

### Optimized prompts (output) should:
- Preserve original intent (don't change what's asked)
- Add 2-4 relevant optimization dimensions
- Be 2-5x longer than input
- Use clear structure (bullet points, numbered steps when appropriate)
- Not over-engineer (don't add all 7 dimensions to every prompt)

## Project Structure

```
prompt-optimizer/
├── data/
│   ├── raw/                # Generated raw pairs
│   ├── train.jsonl         # Training split
│   ├── val.jsonl           # Validation split
│   └── test.jsonl          # Test split
├── scripts/
│   ├── generate_data.py    # Synthetic data generation
│   ├── clean_data.py       # Quality filtering and dedup
│   └── prepare_splits.py   # Train/val/test splitting
├── notebooks/
│   └── finetune_colab.ipynb  # Colab training notebook
├── eval/
│   ├── evaluate.py         # Automated evaluation
│   └── results/            # Evaluation outputs
├── docs/
│   ├── PLAN.md             # This file
│   └── CV_ENTRY.md         # CV description draft
└── README.md               # GitHub/HF README
```

## Timeline

| Phase | Days | Description |
|-------|------|-------------|
| Data design | 1-2 | Categories, criteria, templates |
| Data generation | 1-2 | LLM-based synthetic pairs |
| Data cleaning | 1 | Filtering, dedup, quality check |
| Fine-tuning | 1 | QLoRA on Colab T4 |
| Evaluation | 1 | Baseline vs finetuned comparison |
| Publishing | 1 | HuggingFace model + GitHub repo |
| **Total** | **6-8 days** | |

## CV Entry (Draft)

> **Prompt Optimizer - LLM Fine-Tuning Project**
> Fine-tuned Qwen2.5-3B-Instruct with QLoRA for automatic prompt optimization. Created a synthetic instruction dataset of 2000+ prompt pairs covering 10 categories. The model transforms vague user prompts into structured, effective instructions while preserving original intent. Evaluated against baseline with [X]% improvement in prompt clarity scores. Published model on HuggingFace with full training pipeline on GitHub.
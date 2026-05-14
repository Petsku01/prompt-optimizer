# Prompt Optimizer — LLM Fine-Tuning Project

Fine-tune Qwen2.5-3B-Instruct with QLoRA to automatically transform vague, underspecified prompts into clear, structured, and effective prompts.

## What It Does

| Input (vague) | Output (optimized) |
|---|---|
| write about dogs | Write a 500-word informative essay about working dog breeds, covering their historical roles, modern applications, and training requirements. Use a professional tone and support your claims with specific examples. |
| fix my python code | Debug the following Python code. Identify all bugs and issues, explain the root cause of each, and provide the corrected version with inline comments explaining each fix. Focus on: off-by-one errors, edge cases, and type safety. |
| give me ideas for a mobile app | Generate 10 mobile app ideas for the Finnish market that solve real problems in daily life. For each idea, provide: (1) the problem it solves, (2) target demographic, (3) key differentiator from existing solutions, (4) estimated development complexity (Low/Medium/High). Prioritize ideas that can be built by a solo developer in under 3 months. |

## Dataset

- **1,183 prompt optimization pairs** across 10 categories
- Categories: writing, coding, analysis, translation, Q&A, roleplay, summarization, brainstorming, instruction, editing
- Split: 946 train / 118 val / 119 test
- Format: instruction-tuning (instruction, input, output)

### Data Quality Metrics

| Metric | Value |
|---|---|
| Avg specificity markers | 2.0/7 |
| Avg length ratio (optimized/vague) | 8.3x |
| Has structure | 55% |
| Has format specification | 39% |
| Has audience targeting | 34% |

## Fine-Tuning

- **Base model:** Qwen2.5-3B-Instruct (via Unsloth)
- **Method:** QLoRA (4-bit quantization + LoRA adapters)
- **Hardware:** Google Colab T4 GPU (16GB VRAM)
- **LoRA config:** r=16, alpha=16, dropout=0
- **Training:** 3 epochs, batch=4, grad_accum=4, lr=2e-4
- **Effective batch size:** 16
- **Max sequence length:** 2048

### Quick Start

1. Open `notebooks/finetune_colab.ipynb` in Google Colab
2. Change runtime to **T4 GPU**
3. Upload `data/train.jsonl` and `data/val.jsonl`
4. Run all cells

Expected training time: ~2-4 hours on T4.

## Project Structure

```
prompt-optimizer/
├── data/
│   ├── raw/                    # Generated and seed data
│   ├── train.jsonl             # Training split (946 examples)
│   ├── val.jsonl               # Validation split (118 examples)
│   ├── test.jsonl              # Test split (119 examples)
│   └── cleaned_data.jsonl      # Full cleaned dataset
├── scripts/
│   ├── generate_data_v2.py     # Template-based data generation
│   ├── augment_data.py         # Data augmentation (paraphrases)
│   ├── prepare_final.py        # Cleaning, dedup, splitting
│   └── clean_data_v2.py        # Quality filtering
├── notebooks/
│   └── finetune_colab.ipynb    # Colab training notebook
├── eval/
│   └── evaluate.py             # Evaluation script
├── docs/
│   ├── PLAN.md                 # Project plan
│   └── CV_ENTRY.md             # CV description
└── README.md
```

## Evaluation

```bash
python eval/evaluate.py
```

Metrics: specificity markers, length ratio, format/audience/tone coverage, per-category breakdown.

## References

See [docs/REFERENCES.md](docs/REFERENCES.md) for the full bibliography. Key papers:

- **QLoRA** — Dettmers et al. (2023), our fine-tuning method
- **LoRA** — Hu et al. (2022), adapter-based training foundation
- **Self-Instruct** — Wang et al. (2023), synthetic data generation methodology
- **InstructGPT** — Ouyang et al. (2022), instruction-following paradigm
- **Qwen2.5** — Yang et al. (2024), our base model
- **Automatic Prompt Optimization** — Pryzant et al. (2023), prompt optimization research

## License

MIT

## Author

Petteri Kosonen — Built as a portfolio project demonstrating LLM fine-tuning, dataset creation, and prompt engineering expertise.
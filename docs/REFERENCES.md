# References

## Core Papers

### QLoRA — Efficient Fine-Tuning Method
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. NeurIPS 2023.
  - [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
  - The paper introducing QLoRA: 4-bit NormalFloat quantization, double quantization, and paged optimizers enabling fine-tuning of 65B parameter models on a single 48GB GPU. This is the core technique used in our project.

### LoRA — Low-Rank Adaptation
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022.
  - [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
  - Introduces Low-Rank Adaptation: freezing pre-trained model weights and injecting trainable rank decomposition matrices. Foundational for our adapter-based approach.

### Self-Instruct — Synthetic Data Generation
- Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., & Hajishirzi, H. (2023). **Self-Instruct: Aligning Language Models with Self-Generated Instructions**. ACL 2023.
  - [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
  - Framework for generating instruction-tuning data from LLMs themselves. Our synthetic data generation approach follows similar principles — creating diverse prompt pairs programmatically.

### InstructGPT — Instruction Following via RLHF
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). **Training language models to follow instructions with human feedback**. NeurIPS 2022.
  - [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
  - The InstructGPT paper demonstrating that fine-tuning with human feedback produces models that better follow instructions. Validates the instruction-tuning paradigm our project builds on.

## Model & Framework

### Qwen2.5 — Base Model
- Yang, A., Yang, A., Liu, A., Hoog, L.,... & Zhou, H. (2024). **Qwen2.5 Technical Report**.
  - [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) ([Semantic Scholar](https://www.semanticscholar.org/paper/Qwen2.5-Technical-Report-Yang-Yang/88aa6b1f37d1fd8e0a40499ce9bb87873f03aaa8))
  - Technical report for the Qwen2.5 model family. We use Qwen2.5-3B-Instruct as our base model due to its strong instruction-following capability and multilingual support at a small parameter count.

### Unsloth — Training Framework
- Unsloth AI. **Unsloth: 2-5x Faster LLM Fine-Tuning**.
  - [GitHub](https://github.com/unslothai/unsloth) | [Documentation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) | [Blog: 3x Faster Training](https://unsloth.ai/docs/blog/3x-faster-training-packing)
  - Open-source framework providing optimized CUDA kernels for faster fine-tuning. Used in our training pipeline on Google Colab T4 GPU.

## Prompt Optimization

### Automatic Prompt Optimization
- Pryzant, R., Martinez, R., Fredericks, C., Kennedy, K., Jin, Y., Wany, S., ... & Dai, X. (2023). **Automatic Prompt Optimization with "Gradient Descent" and Beam Search**. EMNLP 2023.
  - [arXiv:2305.03495](https://arxiv.org/abs/2305.03495)
  - Proposes treating prompt optimization as gradient-free search over natural language. Relevant to our project's goal of automatically improving prompt quality.

### Systematic Survey of Prompt Optimization
- Chen, X., et al. (2025). **A Systematic Survey of Automatic Prompt Optimization Techniques**.
  - [arXiv:2502.16923](https://arxiv.org/abs/2502.16923)
  - Comprehensive survey of automatic prompt optimization methods covering gradient-based, gradient-free, and LLM-based approaches.

### DSPy — Declarative Prompt Programming
- Khattab, O., Singhvi, A., Maheshwari, P., ... & Zaharia, M. (2024). **DSPy: The Framework for Programming Language Models**.
  - [arXiv:2604.04869](https://arxiv.org/pdf/2604.04869) | [GitHub](https://github.com/stanfordnlp/dspy)
  - Framework for programmatically optimizing prompts and model pipelines. Complementary approach to our fine-tuning method.

## Synthetic Data & Evaluation

### LLM-Driven Synthetic Data Generation
- Li, J., et al. (2024). **On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey**. ACL Findings 2024.
  - [arXiv:2406.15126](https://arxiv.org/abs/2406.15126) | [ACL Anthology](https://aclanthology.org/2024.findings-acl.658/)
  - Survey on using LLMs for synthetic data generation, curation, and evaluation. Relevant to our template-based data generation pipeline.

### LLM-as-Judge Evaluation
- Zheng, L., Chiang, W. L., Sheng, T., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023). **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**. NeurIPS 2023.
  - [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
  - Proposes using LLMs as judges for evaluating text generation quality. Applicable to evaluating our model's prompt optimization outputs.

### Stanford Alpaca — Instruction-Tuning Data
- Taori, R., et al. (2023). **Stanford Alpaca: An Instruction-following LLaMA Model**.
  - [GitHub](https://github.com/tatsu-lab/stanford_alpaca) | [Project Page](https://crfm.stanford.edu/2023/03/13/alpaca.html)
  - Demonstrated training an instruction-following model from 52K synthetic examples generated by GPT-3.5. Our project follows a similar template-based data generation approach but for prompt optimization specifically.

## Additional Resources

- **HuggingFace PEFT**: [github.com/huggingface/peft](https://github.com/huggingface/peft) — Parameter-Efficient Fine-Tuning methods (LoRA, QLoRA, adapters)
- **TRL (Transformer Reinforcement Learning)**: [github.com/huggingface/trl](https://github.com/huggingface/trl) — SFTTrainer used in our training pipeline
- **Google Colab**: Free T4 GPU access for fine-tuning experiments
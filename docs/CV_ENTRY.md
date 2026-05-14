# CV Entry — Prompt Optimizer Fine-Tuning Project

## English (for international CVs)

### Prompt Optimizer — QLoRA Fine-Tuning for Prompt Engineering

Fine-tuned Qwen2.5-3B-Instruct (3B parameters) with QLoRA (4-bit quantization + LoRA adapters) to automatically transform vague user prompts into structured, effective instructions. Created a synthetic instruction-tuning dataset of 1,200+ prompt pairs across 10 categories (writing, coding, analysis, translation, Q&A, roleplay, summarization, brainstorming, instruction, editing) using template-based generation and augmentation. Model was trained on Google Colab T4 GPU using Unsloth framework. Achieved 8.3x average length improvement with 2.0+ specificity dimensions per optimized prompt. Published model and training pipeline on GitHub.

**Technologies:** PyTorch, Transformers, Unsloth, QLoRA/PEFT, HuggingFace, Python, Google Colab

**Key skills demonstrated:**
- LLM fine-tuning (QLoRA) on consumer hardware
- Synthetic dataset creation and quality filtering
- Instruction-tuning data format design
- Model evaluation (specificity, structure, coverage)
- End-to-end ML project (data → training → evaluation → deployment)

---

## Suomi (suomalaisiin CV:hin)

### Prompt Optimizer — QLoRA-hienosäätö promptien optimointiin

Hienosäädin Qwen2.5-3B-Instruct -mallin (3B parametria) QLoRA-menetelmällä (4-bit kvantisointi + LoRA-adapterit) muuttamaan epämääräisiä käyttäjän promptteja selkeiksi, jäsennellyiksi ohjeiksi. loin synteettisen instruction-tuning -datasetin 1 200+ prompttiparilla 10 kategoriassa (kirjoittaminen, koodaus, analyysi, käännös, Q&A, roolileikki, tiivistäminen, ideointi, ohjeistus, editointi) käyttäen mallipohjaista generointia ja augmentointia. Malli koulutettiin Google Colab T4 GPU:lla Unsloth-kehystä käyttäen. Keskimäärin 8.3-kertainen pituusparannus ja 2.0+ spesifisyysdimensiota optimoidussa promptissa. Malli ja koulutusputki julkaistu GitHubissa.

**Teknologiat:** PyTorch, Transformers, Unsloth, QLoRA/PEFT, HuggingFace, Python, Google Colab

**Keskeiset taidot:**
- LLM-hienosäätö (QLoRA) kuluttajalaitteistolla
- Synteettisen datasetin luonti ja laadunvalvonta
- Instruction-tuning -dataformaatin suunnittelu
- Mallin evaluointi (spesifisyys, rakenne, kate)
- Kokonaisvaltainen ML-projekti (data → koulutus → evaluointi → julkaisu)
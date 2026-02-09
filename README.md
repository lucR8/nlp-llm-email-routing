# LLM-based Customer Support Ticket Routing
The goal of the project is to compare different lightweight Large Language Model (LLM) approaches for **automatic routing of customer support tickets** into predefined departments.

## Project Objective

The task consists in routing customer support tickets into **five departments**:
- Technical Support
- Customer Service
- Billing and Payments
- Sales and Pre-Sales
- General Inquiry

We compare **three routing strategies** on the same dataset split and evaluate them on:
- **Accuracy** (main metric),
- **Inference time**,
- **Memory usage (RAM)**.

## Dataset

We use the Hugging Face dataset:

**`Tobi-Bueck/customer-support-tickets`**

Preprocessing steps:
- Keep only English tickets (`language = "en"`),
- Keep only the five target queues,
- Concatenate `Subject` and `Body` into a single input text,
- Use a fixed and reproducible train / validation / test split.

All preprocessing logic is implemented in `datapreparation.py` and shared by all agents to ensure a fair comparison.

## Compared Agents

### Agent 1 — GPT-2 Prompting (Frozen)
- Model: `distilgpt2`
- Strategy: zero-shot prompting with a frozen model
- No training is performed
- Included as a baseline to highlight the limitations of naive prompting for classification

### Agent 2 — GPT-2 + LoRA
- Model: `distilgpt2`
- Strategy: parameter-efficient fine-tuning with **LoRA (PEFT)**
- A classification head is trained together with LoRA adapters
- Selected checkpoint based on validation performance

### Agent 3 — DistilBERT Classifier
- Model: `distilbert-base-uncased`
- Strategy: supervised discriminative text classification
- Fine-tuned end-to-end for 5-class routing
- Represents a strong task-appropriate baseline

## Repository Structure

```
.
├── notebook.ipynb              # Main experiment notebook (end-to-end pipeline)
├── datapreparation.py          # Dataset loading, filtering, and splitting
├── distilbert_classifier.py    # DistilBERT training and evaluation
├── gpt2_prompting.py           # GPT-2 zero-shot prompting baseline
├── gpt2_lora.py                # GPT-2 + LoRA fine-tuning
├── eval_utils.py               # Evaluation utilities (accuracy, timing, RAM)
├── config.py                   # Centralized paths and configuration
├── tables/
├── figures/
```

## How to Run

1. Install dependencies (recommended: conda or venv)
2. Open and run `notebook.ipynb` from top to bottom
3. The notebook will:
   - Prepare the dataset
   - Evaluate Agent 1 (prompting)
   - Train and evaluate Agent 2 (GPT-2 + LoRA)
   - Train and evaluate Agent 3 (DistilBERT)
   - Save metrics to `summary_results.csv`
   - Generate comparison plots

## Results Summary

The final comparison is stored in:
- `tables/summary_results.csv`
- `figures/comparison_accuracy.png`
- `figures/comparison_time_per_item.png`

Main observations:
- Zero-shot prompting performs poorly for multi-class routing
- LoRA fine-tuning greatly improves GPT-2 with minimal overhead
- DistilBERT achieves the best overall accuracy and remains the most suitable model for this task

## Author

**Luc Renaud**  
LLM Project  
University of Verona (2025–26)

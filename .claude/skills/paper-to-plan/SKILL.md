# Skill: Paper-to-Implementation Plan

## Description
Given a research paper (PDF), produce a comprehensive `plan.md` that serves as a complete blueprint for implementing the paper's core idea in native PyTorch, runnable on a Google Colab T4 GPU (16GB VRAM).

## When to Use
When the user provides a research paper and asks to create an implementation plan, break it down, or prepare it for coding.

## Process

### Phase 1: Deep Paper Read
1. Read the **entire** paper — every page, including appendices and supplementary tables. Do not skim.
2. Extract and note:
   - **Core idea/architecture** — what is the novel contribution?
   - **All mathematical formulas** — every equation that defines a forward pass, loss, or schedule.
   - **Hyperparameters** — every number from every table (model dimensions, learning rates, dropout, etc.).
   - **Training details** — optimizer, LR schedule, regularization, batching strategy, number of steps/epochs.
   - **Evaluation details** — which benchmarks, which metrics, what inference procedure (beam search, sampling, etc.).
   - **Subtle details that are easy to miss** — weight tying, normalization placement (pre vs post), scaling factors, epsilon values, specific activation functions, gradient clipping, etc. These make or break a faithful implementation.

### Phase 2: Dataset Research
The paper's original benchmarks are almost always too large for a Colab T4 demo. Find a suitable substitute:

1. **Identify the task type** from the paper (translation, classification, generation, etc.).
2. **Research small, open-source, readily-available datasets** for the same task using web search:
   - Must be downloadable via `datasets` (HuggingFace), `torchvision`, `torchtext`, or direct URL.
   - Must have train/val/test splits.
   - Must train in **30–60 minutes** on a T4.
   - Prefer datasets with **published baselines** in the paper, so the user can verify their implementation.
3. **Evaluate 3–5 candidates** with concrete numbers:
   - Size (samples/pairs), download method, maintenance status, estimated training time.
4. **Pick one and justify** why it was chosen, and briefly note why others were rejected.

**Common dataset mappings by task:**
| Task | Paper benchmark (too large) | Good Colab substitute |
|---|---|---|
| Machine Translation | WMT (millions of pairs) | IWSLT14 DE-EN (~160K), Multi30k (~29K for smoke tests) |
| Image Classification | ImageNet (1.2M) | CIFAR-10/100 (50K), Tiny-ImageNet (100K) |
| Language Modeling | WikiText-103, C4 | WikiText-2 (2M tokens), PTB (~1M tokens) |
| Object Detection | COCO (118K) | Pascal VOC (5K–11K), COCO subset |
| Text Classification | Large-scale NLP benchmarks | IMDB (25K), SST-2 (67K), AG News (120K) |
| Speech | LibriSpeech (960h) | LibriSpeech-clean-100 or LJSpeech (24h) |
| Summarization | CNN/DailyMail (300K) | XSum subset, SAMSum (16K) |
| Question Answering | SQuAD (100K), Natural Questions | SQuAD (fits T4), TriviaQA subset |

These are starting points — always verify availability via web search, as datasets get taken down or restructured.

### Phase 3: Model Configuration
1. Start with the paper's exact hyperparameters.
2. **Scale down** to fit T4 time/memory budget while keeping the architecture **structurally identical**:
   - Reduce layer count (e.g., 6 → 3)
   - Reduce hidden dimensions (e.g., 512 → 256)
   - Reduce vocabulary size if needed
   - Keep architectural choices the same (activation functions, norm placement, skip connections, etc.)
3. Present both configs in a **side-by-side table**: paper's original vs. our scaled-down version.
4. Estimate parameter count for the scaled-down model.

### Phase 4: Write the Plan
Create `plan.md` with these sections:

#### 1. Paper Summary (2-3 sentences)
What the paper proposes, why it matters.

#### 2. Dataset Choice
Table of properties, justification, rejected alternatives.

#### 3. Model Configuration
Side-by-side hyperparameter table (paper vs. ours).

#### 4. Implementation Plan — Cell-by-Cell Notebook Structure
The notebook must be a **single self-contained `.ipynb` file**, runnable top-to-bottom in Colab.

Organize into numbered cells:
- **Cell 0: Setup & Installs** — pip installs, imports
- **Cell 1: Config** — all hyperparameters in one place, device detection, seed
- **Cell 2: Data Pipeline** — download, tokenize, Dataset class, DataLoader with masks/padding
- **Cells 3–N: Model Components** — one cell per logical component (e.g., positional encoding, attention, FFN, encoder, decoder, full model). For each:
  - Class name and signature
  - Which paper section it implements
  - The exact formula it computes
  - Any subtle details (masking, scaling, dropout placement)
- **Loss cell** — implement the paper's exact loss (label smoothing, etc.)
- **Optimizer & Schedule cell** — exact optimizer config and LR schedule formula
- **Training loop cell** — train function, eval function, epoch count, logging
- **Inference cell** — greedy decoding (minimum), beam search (if paper uses it)
- **Evaluation cell** — compute the paper's metric (BLEU, accuracy, perplexity, etc.) using standard tools
- **Run training cell** — instantiate everything, train, plot loss curves
- **Run evaluation cell** — load best checkpoint, compute metrics, show qualitative examples

#### 5. Key Paper Details to Get Right
A numbered list of **specific, subtle implementation details** that are easy to miss. These are the things that distinguish a faithful implementation from a broken one. Examples:
- Normalization placement (pre-norm vs. post-norm)
- Weight tying schemes
- Scaling factors on embeddings or attention
- Specific epsilon values in optimizers
- Dropout placement (before or after residual add?)
- Activation functions (GeLU vs ReLU vs SiLU)
- Initialization schemes
- Gradient clipping values

#### 6. File Structure
Show the repo layout. Prefer everything in a single notebook.

#### 7. Dependencies
Table of pip packages, their purpose, and whether they're pre-installed in Colab.

#### 8. Estimated Training Time on T4
Breakdown by phase (data prep, training, evaluation, total).

## Output Format
- Output is always a file named `plan.md` in the project root.
- Use markdown tables liberally — they make configs scannable.
- Include code snippets (class signatures, formulas) in the cell descriptions, but do NOT write the full implementation. The plan is a blueprint, not the code.
- Be specific and concrete — "dropout of 0.1 applied after the ReLU in the FFN" not "apply regularization".

## Principles
- **Faithfulness over simplicity**: Every architectural choice in the plan must match the paper. If we scale down, we scale dimensions/layers, never change the fundamental design.
- **Self-contained**: The notebook must run top-to-bottom with zero manual intervention beyond clicking "Run All" in Colab.
- **Colab-first**: All dependencies must be pip-installable. No system packages, no manual downloads, no API keys.
- **Demo not reproduction**: We are NOT trying to reproduce the paper's exact numbers. We want a working, technically correct implementation that trains and produces reasonable (not SOTA) results on a smaller dataset.
- **Single notebook**: No external `.py` files. Everything in one `.ipynb`. This makes it maximally portable and shareable.

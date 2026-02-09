# Skill: Plan-to-Notebook Implementation

## Description
Given a `plan.md` (produced by the `paper-to-plan` skill), generate a single self-contained `.ipynb` Jupyter notebook that faithfully implements the paper in PyTorch, runnable top-to-bottom on a Google Colab T4 GPU.

## When to Use
When the user has a `plan.md` and asks to implement it, build the notebook, or "convert the plan to code."

## Input
- A `plan.md` file in the project root (produced by `paper-to-plan.md` skill).
- The plan contains: paper summary, dataset choice, model config table, cell-by-cell structure, key paper details, dependencies.

## Output
- A single `.ipynb` file (name from the plan's file structure section).
- No external `.py` files. Everything in one notebook.

---

## Process

### Step 1: Read the Plan Thoroughly
Read `plan.md` end-to-end. Pay special attention to:
- The **cell-by-cell structure** — this is your blueprint. Each numbered cell becomes one or more notebook cells.
- The **model configuration table** — use the "our config" column, not the paper column.
- The **"Key Paper Details to Get Right"** section — these are the bugs you'd otherwise introduce. Treat each item as a checklist.
- The **dependencies table** — these go in Cell 0.

### Step 2: Build the Notebook Structure
Create the `.ipynb` as valid JSON with this skeleton:

```
Cell 0:  [markdown] Title + 2-3 line description (paper name, task, architecture summary)
Cell 1:  [code] pip installs (quiet mode: -q flag)
Cell 2:  [code] Imports + Config dataclass + device detection + seed
Cell 3:  [code] Data download + tokenizer training
Cell 4:  [code] Dataset class + DataLoader
Cell 5+: [code] Model components (one per cell, bottom-up order)
         ...    Loss function
         ...    Optimizer + LR schedule
         ...    Training loop (train_epoch + evaluate functions)
         ...    Inference (greedy decode, optionally beam search)
         ...    Metric computation (BLEU, accuracy, perplexity, etc.)
Cell N-1:[code] Run training
Cell N:  [code] Run evaluation + show results
```

### Step 3: Implement Each Cell
Follow the rules below for each cell type.

---

## Cell-by-Cell Implementation Rules

### Markdown Header Cell
- Paper title, arxiv link, 2-3 sentence summary.
- Mention: task, dataset, key architectural choices that make this faithful.

### Cell: Setup & Installs
```python
!pip install <packages> -q
```
- Always use `-q` for quiet install.
- Only install what's NOT pre-installed in Colab (torch, numpy, matplotlib, tqdm are already there).

### Cell: Configuration
- Use a `@dataclass` to hold ALL hyperparameters in one place.
- Include inline comments showing the paper's original value: `n_layers: int = 3  # Paper: 6`.
- Include optimizer params (betas, epsilon) — these often differ from PyTorch defaults.
- Include special token IDs as fields (updated after tokenizer training).
- Set seed for `random`, `numpy`, `torch`, `torch.cuda`.
- Device detection with CPU fallback:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- Print the config at the end so the user can verify.

### Cell: Data Pipeline

**HuggingFace datasets compatibility (CRITICAL):**
- `datasets>=4.0` removed support for `.py` dataset scripts entirely.
- Many older community datasets (e.g., `bbaaaa/iwslt14-de-en`) will throw: `RuntimeError: Dataset scripts are no longer supported`.
- Fix: load from the auto-converted parquet branch:
  ```python
  load_dataset("owner/dataset", "config-name", revision="refs/convert/parquet")
  ```
- You MUST specify the config/subset name explicitly.
- `trust_remote_code=True` does NOT work — it was fully removed.
- Always add a comment explaining why `revision="refs/convert/parquet"` is needed.

**Tokenizer training:**
- If the plan calls for a shared vocabulary (common in NMT), combine source + target text for training.
- Use the HuggingFace `tokenizers` library for BPE — it's fast and Colab-friendly.
- Set up post-processing to auto-add `<bos>` and `<eos>` via `TemplateProcessing`.
- After training, update the config's special token IDs from the actual tokenizer.
- Enable padding and truncation on the tokenizer object itself.
- Print a tokenizer test (encode → tokens → decode) so the user can verify it works.

**Dataset class:**
- Pre-tokenize everything in `__init__` and store as tensor pairs — avoids repeated tokenization.
- Use `tqdm` for the tokenization loop so progress is visible.

**DataLoader:**
- Use `nn.utils.rnn.pad_sequence` in the collate function — don't pad to max_seq_len, pad to the longest in the batch (saves compute).
- `pin_memory=True` when using CUDA.
- `drop_last=True` for training loader (avoids small last batch issues).
- Print batch counts and a sample batch shape as a sanity check.

### Cells: Model Components

**General rules:**
- One logical component per cell (e.g., PositionalEncoding, MultiHeadAttention, FFN, EncoderLayer+Encoder, DecoderLayer+Decoder, full Transformer).
- Order bottom-up: primitives first, compositions last.
- Each cell starts with a comment block citing the paper section and the formula it implements.
- Each cell ends with a `print()` confirming readiness + key dimensions.
- Use the config values, not hardcoded numbers.

**Mask utilities — dtype consistency (CRITICAL):**
- Padding masks from `!=` comparisons produce `bool` tensors.
- Causal masks from `torch.tril(torch.ones(...))` produce `float` tensors by default.
- If you combine them with `&` (bitwise AND), **both must be bool**, otherwise CUDA throws: `NotImplementedError: "bitwise_and_cuda" not implemented for 'Float'`.
- Fix: explicitly create causal masks as bool:
  ```python
  torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))
  ```
- In `masked_fill`, use `mask == 0` (works for both bool False and float 0.0) or `~mask` for bool masks.

**Attention:**
- Scale by `1/sqrt(d_k)` — not `d_model`, specifically `d_k = d_model / n_heads`.
- Implement attention dropout (after softmax, before matmul with V).
- Mask with `-inf` before softmax, not after.

**Residual connections — norm placement:**
- Post-norm (original Transformer): `LayerNorm(x + Dropout(Sublayer(x)))` — norm AFTER residual add.
- Pre-norm (GPT-2 style): `x + Sublayer(LayerNorm(x))` — norm BEFORE sublayer.
- The plan specifies which one. Get it right — this is the #1 source of implementation bugs.

**Embedding scaling:**
- If the plan says to scale embeddings by `sqrt(d_model)`, do it: `self.embedding(x) * math.sqrt(self.d_model)`.
- This is critical — without it, embedding magnitudes are too small relative to positional encodings.

**Weight tying:**
- Assign weight references BEFORE calling `_init_parameters()`, or the init will create separate weight values that later get overwritten.
- Actually, the safer pattern is: tie weights first, then init. Xavier init on the shared weight will apply once.
  ```python
  self.decoder.embedding.weight = self.encoder.embedding.weight
  self.output_projection.weight = self.encoder.embedding.weight
  self._init_parameters()
  ```

**Parameter initialization:**
- Xavier uniform for all parameters with dim > 1 is the standard default:
  ```python
  for p in self.parameters():
      if p.dim() > 1:
          nn.init.xavier_uniform_(p)
  ```

### Cell: Loss Function
- If the plan specifies label smoothing, implement it manually — PyTorch's `CrossEntropyLoss(label_smoothing=...)` exists but implementing it manually gives more control and matches the paper exactly.
- Always mask out padding positions from the loss.
- Average over non-pad tokens, not over all positions.

### Cell: Optimizer & LR Schedule
- Use `lr=1.0` as the base learning rate when using `LambdaLR` — the lambda function computes the actual LR.
- Guard against step=0 in the lambda: `step = max(step, 1)`.
- Visualize the LR schedule with matplotlib so the user can verify the warmup/decay shape.

### Cell: Training Loop
- Teacher forcing: `tgt_input = tgt[:, :-1]`, `tgt_output = tgt[:, 1:]`.
- Flatten for loss: `logits.view(-1, vocab_size)`, `target.view(-1)`.
- Include gradient clipping (`clip_grad_norm_`, max_norm=1.0) for stability.
- Step the scheduler per step (not per epoch) if the plan's schedule is step-based.
- Track loss weighted by number of non-pad tokens for accurate epoch loss.
- Use `tqdm` with periodic `set_postfix` updates (loss, lr, tokens/sec).
- The `evaluate` function should be decorated with `@torch.no_grad()`.

### Cell: Inference (Greedy Decode)
- Encode source ONCE, then decode autoregressively.
- Start with `<bos>`, append argmax token each step, stop at `<eos>` or max_len.
- Also provide a `translate_sentence` helper that takes raw text → tokenize → decode → detokenize.

### Cell: Inference (Beam Search, if in plan)
- Encode source once, maintain beam_size hypotheses.
- Length normalization: `score / (length ** alpha)`.
- Keep completed beams separate from active beams.

### Cell: Metric Computation
- Use standard metric libraries (`sacrebleu` for BLEU, `sklearn` for accuracy, etc.).
- Decode predictions and references back to text before computing metrics.
- Filter out special tokens (`<pad>`, `<bos>`, `<eos>`) from reference IDs before decoding.

### Cell: Run Training
- Print epoch summary: train loss, val loss, time, LR, and a `*` marker for best val loss.
- Save best model checkpoint based on val loss.
- Show a sample output every N epochs (e.g., a translation, a generated sentence).
- Plot train/val loss curves with matplotlib at the end.

### Cell: Run Evaluation
- Load best checkpoint.
- Compute metrics on the test set.
- Print quantitative results.
- Show 10 qualitative examples (source → reference → prediction).

---

## Common Pitfalls & Fixes

### 1. HuggingFace Dataset Script Error
**Error:** `RuntimeError: Dataset scripts are no longer supported`
**Fix:** `load_dataset("id", "config", revision="refs/convert/parquet")`

### 2. Mask Dtype Mismatch on CUDA
**Error:** `NotImplementedError: "bitwise_and_cuda" not implemented for 'Float'`
**Fix:** Create causal mask with `dtype=torch.bool`.

### 3. Embedding Scale Forgotten
**Symptom:** Model trains but produces garbage — positional encoding dominates.
**Fix:** `self.embedding(x) * math.sqrt(self.d_model)`.

### 4. Wrong Norm Placement
**Symptom:** Model trains but converges to worse loss than expected.
**Fix:** Check plan — post-norm = `LayerNorm(x + sublayer(x))`, pre-norm = `x + sublayer(LayerNorm(x))`.

### 5. Scheduler Stepped Per-Epoch Instead of Per-Step
**Symptom:** LR warmup takes 4000 epochs instead of 4000 steps.
**Fix:** Call `scheduler.step()` inside the batch loop, not the epoch loop.

### 6. Teacher Forcing Off-by-One
**Symptom:** Model sees the answer during training (data leakage).
**Fix:** Input = `tgt[:, :-1]` (without last), Target = `tgt[:, 1:]` (without first/BOS).

### 7. Weight Tying Undone by Init
**Symptom:** Tied weights diverge during training.
**Fix:** Tie weights by assigning `.weight` references; ensure init doesn't create new tensors.

### 8. Attention Mask Applied After Softmax
**Symptom:** Model attends to padding/future positions.
**Fix:** Apply mask BEFORE softmax using `masked_fill(mask == 0, float('-inf'))`.

---

## Quality Checklist

Before delivering the notebook, verify:

- [ ] Every cell runs without error when executed top-to-bottom
- [ ] Config dataclass has inline comments showing paper values
- [ ] Dataset loads successfully (parquet branch if needed)
- [ ] Tokenizer test prints encode → tokens → decode round-trip
- [ ] All masks are bool dtype (no float/bool mixing)
- [ ] Embedding scaling by sqrt(d_model) is present
- [ ] Norm placement matches plan (post-norm vs pre-norm)
- [ ] Weight tying is active (check with `model.encoder.embedding.weight is model.output_projection.weight`)
- [ ] Optimizer eps/betas match the paper, not PyTorch defaults
- [ ] LR schedule shape is correct (visualized in a plot)
- [ ] Teacher forcing split is correct (input = tgt[:-1], target = tgt[1:])
- [ ] Loss ignores padding positions
- [ ] Gradient clipping is present
- [ ] Training cell prints loss, LR, and periodic sample outputs
- [ ] Eval cell loads best checkpoint, computes metrics, shows examples
- [ ] Every code cell ends with a print statement confirming readiness

## Principles
- **Faithfulness over elegance**: Match the plan exactly. Don't "improve" the architecture.
- **One cell, one concern**: Each cell does one thing. The user should be able to re-run individual cells.
- **Print everything**: Every cell should produce visible output — shapes, counts, confirmations. Silent cells make debugging impossible.
- **Fail fast**: Add shape checks and assertions early. A crash at model creation is better than NaN at epoch 10.
- **Colab-first**: No local paths, no system deps, no API keys. `pip install` and go.

# Skill: Plan-to-Notebook Implementation

## Description
Given a `plan.md` (produced by the `paper-to-plan` skill), generate a single self-contained `.ipynb` Jupyter notebook that faithfully implements the paper in PyTorch, runnable top-to-bottom on a Google Colab T4 GPU. Works for any domain — NLP, vision, audio, RL, etc.

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
- The **"Key Paper Details to Get Right"** section — these are the bugs you'd otherwise introduce. Treat each item as a mandatory checklist.
- The **dependencies table** — these go in the install cell.

### Step 2: Build the Notebook Structure
Create the `.ipynb` as valid JSON with this skeleton:

```
Cell 0:  [markdown] Title + 2-3 line description (paper name, task, architecture summary)
Cell 1:  [code] pip installs (quiet mode: -q flag)
Cell 2:  [code] Imports + Config dataclass + device detection + seed
Cell 3:  [code] Data download + preprocessing
Cell 4:  [code] Dataset class + DataLoader
Cell 5+: [code] Model components (one per cell, bottom-up order)
         ...    Loss function
         ...    Optimizer + LR schedule
         ...    Training loop (train_epoch + evaluate functions)
         ...    Inference / generation / prediction
         ...    Metric computation
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
- Only install what's NOT pre-installed in Colab. Pre-installed: `torch`, `torchvision`, `torchaudio`, `numpy`, `matplotlib`, `tqdm`, `PIL`, `sklearn`.

### Cell: Configuration
- Use a `@dataclass` to hold ALL hyperparameters in one place.
- Include inline comments showing the paper's original value: `n_layers: int = 3  # Paper: 6`.
- Include optimizer params (betas, epsilon) — these often differ from PyTorch defaults.
- Set seed for `random`, `numpy`, `torch`, `torch.cuda`.
- Device detection with CPU fallback:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- Print the config at the end so the user can verify.

### Cell: Data Pipeline
- Follow whatever the plan prescribes for data loading — HuggingFace `datasets`, `torchvision.datasets`, direct URL download, etc.
- If the plan says to train a tokenizer, build a custom transform, or preprocess images — implement exactly as specified.
- Print dataset sizes, a sample, and sanity-check shapes after loading.

**DataLoader best practices:**
- For variable-length sequences, use a collate function with dynamic padding (pad to longest in batch, not max_seq_len).
- `pin_memory=True` when using CUDA.
- `drop_last=True` for training loader (avoids small last-batch edge cases).
- Print batch counts and a sample batch shape as a sanity check.

### Cells: Model Components

**General rules:**
- One logical component per cell, ordered bottom-up: primitives first, compositions last.
- Each cell starts with a comment block citing the paper section and the formula it implements.
- Each cell ends with a `print()` confirming readiness + key dimensions.
- Use the config values, not hardcoded numbers.

**Follow the plan's "Key Paper Details" exactly.** Common detail categories:
- **Normalization placement**: post-norm (`LayerNorm(x + Sublayer(x))`) vs pre-norm (`x + Sublayer(LayerNorm(x))`) — the #1 source of implementation bugs.
- **Scaling factors**: embedding × √d_model, attention ÷ √d_k, etc. Missing these causes training to diverge or produce garbage.
- **Weight tying**: if the plan specifies shared weights, assign `.weight` references before calling parameter init.
- **Dropout placement**: before vs after residual add produces different results. Match the plan.
- **Activation functions**: ReLU vs GeLU vs SiLU vs Swish — don't substitute.
- **Initialization**: Xavier, Kaiming, normal with specific std — match what the plan says.

**Parameter initialization (default):**
```python
for p in self.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```

### Cell: Loss Function
- Implement the paper's exact loss as specified in the plan (cross-entropy, label smoothing, contrastive, reconstruction, etc.).
- If there are ignored classes (e.g., padding), mask them out.
- Average over valid elements, not over all positions.

### Cell: Optimizer & LR Schedule
- If using `LambdaLR`, set `lr=1.0` as the base — the lambda computes the actual LR.
- Guard against step=0: `step = max(step, 1)`.
- Match the paper's exact optimizer params (betas, epsilon, weight decay) — don't rely on PyTorch defaults.
- Visualize the LR schedule with matplotlib so the user can verify the shape.

### Cell: Training Loop
- Implement `train_epoch()` and `evaluate()` functions.
- For seq2seq with teacher forcing: `input = tgt[:, :-1]`, `target = tgt[:, 1:]`.
- Flatten for loss when needed: `logits.view(-1, n_classes)`, `target.view(-1)`.
- Include gradient clipping (`clip_grad_norm_`, max_norm=1.0) for stability.
- Step the scheduler per-step or per-epoch — match what the plan's LR formula requires.
- Use `tqdm` with periodic `set_postfix` updates (loss, lr, throughput).
- `evaluate()` must use `@torch.no_grad()`.

### Cell: Inference
- Implement whatever inference procedure the plan specifies (greedy decode, beam search, sampling, single forward pass, etc.).
- Provide a user-friendly helper that takes raw input → preprocess → model → postprocess → readable output.

### Cell: Metric Computation
- Use standard metric libraries as specified in the plan (`sacrebleu`, `sklearn.metrics`, `torchmetrics`, etc.).
- Postprocess model outputs to match metric input format (detokenize, denormalize, etc.).

### Cell: Run Training
- Print epoch summary: train loss, val loss, time, LR, `*` marker for best val loss.
- Save best model checkpoint based on val loss.
- Show a sample output every N epochs so the user can watch progress qualitatively.
- Plot train/val loss curves with matplotlib at the end.

### Cell: Run Evaluation
- Load best checkpoint.
- Compute metrics on the test set.
- Print quantitative results.
- Show qualitative examples (8-10 samples: input → reference → prediction).

---

## Known Runtime Errors & Fixes

These are real errors encountered on Colab that will silently waste the user's time if not avoided.

### 1. HuggingFace Dataset Script Error
```
RuntimeError: Dataset scripts are no longer supported
```
**Cause:** `datasets>=4.0` dropped support for `.py` dataset loading scripts. Many older community datasets still have them.
**Fix:** Load from the auto-converted parquet branch:
```python
load_dataset("owner/dataset", "config-name", revision="refs/convert/parquet")
```
- You MUST specify the config/subset name explicitly.
- `trust_remote_code=True` does NOT work — it was fully removed in datasets 4.0.
- Always add a comment explaining why this is needed.

### 2. Bitwise Ops on Mixed dtypes (CUDA)
```
NotImplementedError: "bitwise_and_cuda" not implemented for 'Float'
```
**Cause:** Combining a `bool` tensor (e.g., padding mask from `!=`) with a `float` tensor (e.g., causal mask from `torch.tril(torch.ones(...))`) using `&`. Works on CPU, crashes on CUDA.
**Fix:** Always create all masks with consistent dtype. Use `dtype=torch.bool` explicitly:
```python
torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))
```

### 3. CUDA OOM During Evaluation
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Cause:** Evaluation or inference accumulating gradients, or generating very long sequences.
**Fix:**
- Wrap eval/inference in `@torch.no_grad()` or `with torch.no_grad():`.
- Process test set in batches, not all at once.
- For autoregressive generation, set a reasonable `max_len`.

### 4. DataLoader Multiprocessing Crash
```
RuntimeError: DataLoader worker (pid ...) exited unexpectedly
```
**Cause:** `num_workers > 0` combined with large in-memory datasets in Colab.
**Fix:** Use `num_workers=2` (not higher) on Colab. If it still crashes, fall back to `num_workers=0`.

### 5. NaN Loss After a Few Steps
**Cause:** Usually one of:
- Missing embedding scaling (positional encoding dominates).
- Wrong LR schedule (too high initial LR if warmup is misconfigured).
- Mask applied after softmax instead of before.
**Fix:** Check the plan's "Key Paper Details" section — the fix is almost always there.

### 6. Model Produces Constant/Repeated Output
**Cause:** Usually one of:
- Weight tying was specified but not implemented (or undone by init).
- Scheduler stepped per-epoch when it should be per-step (warmup never completes).
- Label smoothing distributing probability to padding index.
**Fix:** Verify weight tying with `param_a is param_b`, check scheduler via the LR plot, ensure loss masks padding.

---

## Quality Checklist

Before delivering the notebook, verify:

- [ ] Valid `.ipynb` JSON — parseable by `json.load()`
- [ ] Every cell runs without error when executed top-to-bottom
- [ ] Config dataclass has inline comments showing paper values vs. our values
- [ ] Dataset loads successfully (parquet branch workaround if needed)
- [ ] Data sanity check printed (sizes, sample, shapes)
- [ ] All tensor masks use consistent dtype (all bool or all float, never mixed)
- [ ] Model components match the plan's formulas and paper sections
- [ ] Optimizer params (eps, betas, weight_decay) match the paper, not PyTorch defaults
- [ ] LR schedule visualized in a plot and shape matches expected (warmup, decay, cosine, etc.)
- [ ] Loss function handles ignored/padding positions correctly
- [ ] Gradient clipping is present for training stability
- [ ] Training cell prints loss, LR, and periodic sample outputs
- [ ] Eval cell loads best checkpoint, computes metrics, shows qualitative examples
- [ ] Every code cell ends with a print statement confirming readiness or showing output

---

## Principles
- **Faithfulness over elegance**: Match the plan exactly. Don't "improve" the architecture, swap components, or add features the plan doesn't call for.
- **One cell, one concern**: Each cell does one thing. The user should be able to re-run individual cells during debugging.
- **Print everything**: Every cell should produce visible output — shapes, counts, confirmations. Silent cells make debugging impossible in a notebook.
- **Fail fast**: Add shape checks and assertions early. A crash at model creation is better than NaN at epoch 10.
- **Colab-first**: No local file paths, no system-level packages, no API keys. `pip install` and go.
- **Config is king**: Every hyperparameter lives in the config dataclass. No magic numbers scattered through the code.

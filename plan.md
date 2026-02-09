# Implementation Plan: "Attention Is All You Need" (Original Transformer)

## Paper Summary

The paper introduces the **Transformer**, the first sequence transduction model based entirely on attention, replacing recurrence and convolutions. It uses an encoder-decoder architecture where both sides are stacks of self-attention and feed-forward layers. The model achieves state-of-the-art results on EN-DE and EN-FR machine translation (WMT 2014).

---

## Dataset Choice

**IWSLT 2014 German-English** (`bbaaaa/iwslt14-de-en` on HuggingFace)

| Property | Value |
|---|---|
| Task | German → English translation |
| Train pairs | ~160,000 |
| Val pairs | ~887 |
| Test pairs | ~4,698 |
| Download | `datasets.load_dataset("bbaaaa/iwslt14-de-en")` |
| Download size | ~12.6 MB |

**Why this dataset:**
- The paper's primary benchmark is WMT 2014 EN-DE (4.5M pairs) — far too large for a T4 demo
- IWSLT14 DE-EN is the standard small-scale NMT benchmark used in fairseq and academic papers
- ~160K pairs trains in 30–60 min on a T4 GPU, enough for the model to learn meaningful translations
- Has proper train/val/test splits with published baselines (~35 BLEU for well-tuned small Transformers)
- Same language pair (German ↔ English), faithful to the paper's focus

**Rejected alternatives:**
- Multi30k (29K pairs): too small, model overfits immediately
- WMT 2014 (4.5M+): way too large for Colab
- OPUS-100 (1M pairs): too large, would need hours
- Tatoeba: no standard splits, uneven quality

---

## Model Configuration

We use a **scaled-down** version of the base Transformer to fit within T4 time/memory constraints while keeping the architecture identical.

| Hyperparameter | Paper (base) | Our config |
|---|---|---|
| `N` (layers) | 6 | 3 |
| `d_model` | 512 | 256 |
| `d_ff` | 2048 | 512 |
| `h` (heads) | 8 | 4 |
| `d_k = d_v` | 64 | 64 |
| `P_drop` | 0.1 | 0.1 |
| `ε_ls` (label smoothing) | 0.1 | 0.1 |
| Warmup steps | 4000 | 4000 |
| Max sequence length | — | 128 |
| Params (approx) | 65M | ~10M |

All architectural choices (residual connections, layer norm placement, weight tying, sinusoidal PE, etc.) remain **identical** to the paper.

---

## Implementation Plan — Single Notebook (`transformer.ipynb`)

The notebook will have the following sections, each in its own cell group:

### Cell 0: Setup & Installs
```
pip install datasets tokenizers sacrebleu
```
- `datasets`: HuggingFace datasets for downloading IWSLT14
- `tokenizers`: HuggingFace tokenizers for BPE (matching paper's use of BPE)
- `sacrebleu`: standard BLEU evaluation (matching paper's evaluation metric)

### Cell 1: Configuration
- Single dataclass/dict holding all hyperparameters
- Device detection (CUDA / CPU fallback)
- Seed for reproducibility

### Cell 2: Data Pipeline
1. **Download** IWSLT14 DE-EN via `datasets.load_dataset("bbaaaa/iwslt14-de-en")`
2. **Train a shared BPE tokenizer** using `tokenizers.ByteLevelBPETokenizer` on the combined DE+EN training text
   - Vocab size: ~10,000 (paper used 37K shared BPE; we use smaller to match dataset size)
   - Add special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`
   - Paper detail: shared source-target vocabulary (Section 5.1) — we replicate this
3. **Tokenize** all splits (train/val/test) with the trained BPE
4. **Build a PyTorch Dataset** class that:
   - Returns (src_ids, tgt_ids) pairs
   - Truncates to `max_seq_len=128`
   - Adds `<bos>` and `<eos>` tokens
5. **Build a DataLoader** with:
   - Dynamic batching by approximate sequence length (paper: "batched by approximate sequence length")
   - Collate function that pads to the longest sequence in the batch
   - Returns `src`, `tgt`, `src_mask`, `tgt_mask`

### Cell 3: Model — Positional Encoding (Section 3.5)
```python
class PositionalEncoding(nn.Module)
```
- Sinusoidal positional encoding: `PE(pos,2i) = sin(pos/10000^(2i/d_model))`, `PE(pos,2i+1) = cos(pos/10000^(2i/d_model))`
- Precomputed buffer of shape `(max_len, d_model)`, registered as a non-learnable buffer
- Added to input embeddings, then dropout applied
- Paper detail: embeddings are scaled by `√d_model` before adding PE (Section 3.4)

### Cell 4: Model — Multi-Head Attention (Section 3.2)
```python
class MultiHeadAttention(nn.Module)
```
**Scaled Dot-Product Attention (Section 3.2.1):**
- `Attention(Q, K, V) = softmax(QK^T / √d_k) V`
- Optional mask parameter for:
  - Padding mask (encoder and decoder)
  - Causal/lookahead mask (decoder self-attention only)
- Mask sets positions to `-inf` before softmax

**Multi-Head Attention (Section 3.2.2):**
- Linear projections: `W_Q`, `W_K`, `W_V` each `(d_model, d_k)` per head, implemented as single `(d_model, d_model)` matrices reshaped to `(batch, h, seq_len, d_k)`
- `W_O` output projection: `(h * d_v, d_model)`
- Dropout on attention weights (paper Section 5.4)

**Three uses (Section 3.2.3):**
1. Encoder self-attention: Q=K=V from encoder
2. Decoder masked self-attention: Q=K=V from decoder, with causal mask
3. Encoder-decoder attention: Q from decoder, K=V from encoder output

### Cell 5: Model — Feed-Forward Network (Section 3.3)
```python
class PositionwiseFeedForward(nn.Module)
```
- `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`
- Inner dimension `d_ff` (512 in our config, 2048 in paper)
- ReLU activation
- Dropout after ReLU (from residual dropout description)

### Cell 6: Model — Encoder Layer & Encoder Stack (Section 3.1)
```python
class EncoderLayer(nn.Module)
class Encoder(nn.Module)
```
**EncoderLayer:**
- Sub-layer 1: Multi-Head Self-Attention
- Sub-layer 2: Position-wise FFN
- Each sub-layer wrapped with: residual connection + layer norm
- Paper formula: `LayerNorm(x + Sublayer(x))` — note this is **post-norm** (norm after residual add)
- Dropout on each sub-layer output before the residual add (Section 5.4)

**Encoder:**
- Input embedding (`nn.Embedding`) scaled by `√d_model`
- Add positional encoding
- Dropout on the sum
- Stack of N `EncoderLayer`s

### Cell 7: Model — Decoder Layer & Decoder Stack (Section 3.1)
```python
class DecoderLayer(nn.Module)
class Decoder(nn.Module)
```
**DecoderLayer:**
- Sub-layer 1: **Masked** Multi-Head Self-Attention (causal mask prevents attending to future positions)
- Sub-layer 2: Multi-Head Encoder-Decoder Attention (Q from decoder, K/V from encoder)
- Sub-layer 3: Position-wise FFN
- Same residual + layer norm wrapping as encoder

**Decoder:**
- Output embedding (shared weight matrix with encoder embedding — Section 3.4)
- Add positional encoding
- Dropout on the sum
- Stack of N `DecoderLayer`s

### Cell 8: Model — Full Transformer (Section 3)
```python
class Transformer(nn.Module)
```
- Combines Encoder + Decoder
- Final linear projection to vocab size (pre-softmax)
- **Weight tying** (Section 3.4): the same weight matrix is shared between:
  1. Encoder input embedding
  2. Decoder input embedding
  3. Pre-softmax linear layer
- This is a key paper detail — the embedding weights are shared and the linear layer reuses `embedding.weight`

**Mask generation utilities:**
- `create_padding_mask(seq, pad_idx)` → `(batch, 1, 1, seq_len)` boolean mask
- `create_causal_mask(size)` → `(1, 1, size, size)` upper-triangular mask
- Combined mask for decoder: padding mask AND causal mask

### Cell 9: Loss — Label-Smoothed Cross-Entropy (Section 5.4)
```python
class LabelSmoothingLoss(nn.Module)
```
- Standard cross-entropy but with label smoothing `ε_ls = 0.1`
- True label gets probability `1 - ε_ls`, remaining probability `ε_ls` distributed uniformly across all other classes
- Ignores padding index
- Paper notes: "hurts perplexity but improves accuracy and BLEU"

### Cell 10: Optimizer & LR Schedule (Section 5.3)
```python
class TransformerLRScheduler
```
- Adam optimizer with `β_1 = 0.9`, `β_2 = 0.98`, `ε = 10^-9`
- Learning rate formula: `lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))`
- Linearly increases LR for `warmup_steps` (4000), then decays proportionally to `1/√step`
- Implemented as a `torch.optim.lr_scheduler.LambdaLR`

### Cell 11: Training Loop
```python
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device)
def evaluate(model, dataloader, criterion, device)
```
- Train for **20–30 epochs** (sufficient for IWSLT14 with this model size)
- Per-step: forward pass → loss → backward → gradient clip (optional, not in paper but helps stability) → optimizer step → scheduler step
- Log every N steps: step, loss, learning rate, tokens/sec
- End-of-epoch: run validation, log val loss
- Track best validation loss, save best checkpoint

### Cell 12: Inference — Greedy Decoding
```python
def greedy_decode(model, src, src_mask, max_len, bos_idx, eos_idx)
```
- Encode source sequence once
- Autoregressively generate target tokens: start with `<bos>`, feed through decoder, take argmax, append, repeat until `<eos>` or `max_len`
- This is the simple version; the paper uses beam search but greedy is sufficient for a demo

### Cell 13: Inference — Beam Search (Optional, for completeness)
```python
def beam_search_decode(model, src, src_mask, max_len, bos_idx, eos_idx, beam_size=4, alpha=0.6)
```
- Paper uses beam size 4 with length penalty `α = 0.6`
- Maintain `beam_size` hypotheses at each step
- Length normalization: score divided by `(length^α)`
- This is optional but demonstrates the paper's actual inference procedure

### Cell 14: BLEU Evaluation
```python
def compute_bleu(model, test_dataloader, tokenizer, device)
```
- Translate all test set source sentences using greedy/beam decode
- Detokenize predictions and references using the BPE tokenizer's decode
- Compute corpus-level BLEU using `sacrebleu` (standard tool)
- Print sample translations side by side for qualitative inspection

### Cell 15: Run Training
- Instantiate model, optimizer, scheduler, loss
- Print model parameter count
- Train for configured number of epochs
- Plot training/validation loss curves (matplotlib)

### Cell 16: Run Evaluation & Show Results
- Load best checkpoint
- Run BLEU evaluation on test set
- Print BLEU score
- Show 10 example translations: source (DE) | reference (EN) | predicted (EN)
- Expected BLEU: ~25–30 with our small model on IWSLT14 (published small Transformer baselines get ~34-35)

---

## Key Paper Details to Get Right

These are easy to miss and critical for a faithful implementation:

1. **Post-norm** (not pre-norm): `LayerNorm(x + Sublayer(x))` — the paper uses post-layer-norm, where the norm is applied after the residual add. Many modern implementations use pre-norm instead, but the paper is explicitly post-norm.

2. **Embedding scale factor**: Multiply embedding weights by `√d_model` (Section 3.4). Without this, the embedding magnitudes are too small relative to the positional encodings.

3. **Weight tying three-way**: Encoder embedding, decoder embedding, and pre-softmax linear layer all share the same weight matrix (Section 3.4).

4. **Residual dropout**: Dropout is applied to the output of each sub-layer *before* it is added to the residual (Section 5.4). Additionally, dropout is applied to the sum of embeddings + positional encodings.

5. **Attention dropout**: Dropout applied to the attention weights (after softmax, before multiplying by V).

6. **Label smoothing**: `ε_ls = 0.1` — distributes 10% of probability mass uniformly across the vocabulary. This is not just temperature scaling; it changes the target distribution.

7. **Adam epsilon**: The paper uses `ε = 10^-9`, not PyTorch's default of `10^-8`.

8. **Causal mask in decoder**: Upper-triangular mask of `-inf` applied before softmax in decoder self-attention to prevent attending to future tokens.

9. **Shared BPE vocabulary**: The paper trains a joint source-target BPE vocabulary (Section 5.1).

---

## File Structure

```
paper_to_nb/
├── transformer_og.pdf        # The paper
├── plan.md                   # This plan
└── transformer.ipynb         # Complete implementation notebook
```

Everything in a single notebook — no external Python files. The notebook is self-contained and runnable top-to-bottom in Google Colab with a T4 runtime.

---

## Dependencies (all pip-installable, Colab-compatible)

| Package | Purpose | Pre-installed in Colab? |
|---|---|---|
| `torch` | Model, training | Yes |
| `datasets` | Download IWSLT14 | No → `pip install datasets` |
| `tokenizers` | Train BPE tokenizer | No → `pip install tokenizers` |
| `sacrebleu` | BLEU evaluation | No → `pip install sacrebleu` |
| `matplotlib` | Loss plots | Yes |
| `tqdm` | Progress bars | Yes |

---

## Estimated Training Time on T4

| Phase | Time |
|---|---|
| Data download + BPE training | ~2 min |
| Training (20–30 epochs) | ~30–45 min |
| Test set BLEU evaluation | ~2–3 min |
| **Total** | **~35–50 min** |

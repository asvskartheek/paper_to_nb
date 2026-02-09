# 📄 Paper to Notebook

Convert research papers (PDFs) into runnable implementation Jupyter notebooks using an AI-powered two-stage pipeline.

> **Bring Your Own Key** — you supply your Anthropic API key, pay only for what you use, and your key is never stored.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

---

## ✨ What It Does

Upload a research paper PDF and get back a self-contained `.ipynb` notebook that implements the paper's core ideas — ready to run on Google Colab with a T4 GPU.

The pipeline runs in two stages, each powered by [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via the **Claude Agent SDK**:

```
PDF Upload  →  [Agent 1: Paper-to-Plan]  →  plan.md
            →  [Agent 2: Plan-to-Notebook]  →  output.ipynb  →  Download
```

| Stage | Agent | What it produces |
|-------|-------|------------------|
| **1. Paper → Plan** | Reads the PDF, extracts key concepts, and writes a detailed cell-by-cell implementation plan | `plan.md` |
| **2. Plan → Notebook** | Reads `plan.md` and generates a complete, runnable Jupyter notebook | `output.ipynb` |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **Claude Code CLI** — install globally:
  ```bash
  npm install -g @anthropic-ai/claude-code
  ```
- An **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com/settings/keys)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/paper_to_nb.git
cd paper_to_nb

# Install dependencies with uv
uv sync
```

### Run the Gradio App

```bash
uv run app.py
```

Open the URL printed in the terminal (default: `http://127.0.0.1:7860`) and:

1. Paste your Anthropic API key (`sk-ant-...`).
2. Upload a research paper PDF (up to 50 MB).
3. Click **🚀 Generate Notebook**.
4. Wait for both stages to complete (roughly 5–20 minutes total).
5. Download the generated `.ipynb`.

---

## 🏗️ Project Structure

```
paper_to_nb/
├── app.py                  # Gradio web application (main entry point)
├── main.py                 # Placeholder CLI entry point
├── plan.md                 # Example generated plan (Attention Is All You Need)
├── plan_gradio.md          # Design document for the Gradio interface
├── pyproject.toml          # Project metadata & dependencies
├── transformer.ipynb       # Example generated notebook
├── README.md               # ← You are here
└── workspaces/             # Created at runtime — per-session temp directories
    └── <uuid>/
        ├── paper.pdf       # Uploaded PDF
        ├── plan.md         # Generated plan
        └── output.ipynb    # Generated notebook
```

---

## ⚙️ How It Works

### Agent 1 — Paper-to-Plan

- **System prompt**: Act as an ML researcher; read the paper, produce an actionable implementation plan.
- **Tools**: `Read`, `Write`, `Glob`, `Grep`
- **Max turns**: 30
- **Output**: `plan.md` containing a paper summary, scaled-down model configuration, dataset selection, and cell-by-cell notebook outline.

### Agent 2 — Plan-to-Notebook

- **System prompt**: Act as an ML engineer; convert the plan into a valid `.ipynb` notebook.
- **Tools**: `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `NotebookEdit`
- **Max turns**: 50
- **Output**: `output.ipynb` — a complete, self-contained Jupyter notebook targeting Google Colab (T4 GPU).

Both agents use `permission_mode="acceptEdits"` so they can write files without manual confirmation.

### Session Isolation

Each request creates an isolated workspace under `workspaces/<uuid>/`. The uploaded PDF, generated plan, and notebook are all scoped to that directory, so concurrent users don't interfere with each other.

### API Key Handling

- Your Anthropic API key is passed **in-memory only** via environment variables to each `query()` call.
- It is **never logged, stored to disk, or persisted** in any way.
- The key lives only for the duration of the request.

---

## 🧩 Dependencies

| Package | Purpose |
|---------|---------|
| [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/) (≥ 0.1.33) | Python SDK for Claude Code agent orchestration |
| [`gradio`](https://www.gradio.app/) (≥ 5.0.0) | Web UI for file upload, status display, and download |

Defined in [pyproject.toml](pyproject.toml).

---

## 📖 Example

The repository includes an example run on the **"Attention Is All You Need"** Transformer paper:

- [plan.md](plan.md) — Generated implementation plan (scaled-down config for IWSLT 2014 DE→EN).
- [transformer.ipynb](transformer.ipynb) — Generated notebook implementing a mini Transformer from scratch.

---

## 🛠️ Configuration

Key constants in [app.py](app.py) you can tune:

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_PDF_SIZE_MB` | 50 | Maximum upload size in MB |
| `MAX_PLAN_TURNS` | 30 | Max agent turns for plan generation |
| `MAX_NOTEBOOK_TURNS` | 50 | Max agent turns for notebook generation |
| `WORKSPACES_DIR` | `./workspaces` | Directory for per-session workspaces |

---

## ⚠️ Limitations & Notes

- **Cost**: Each run consumes Anthropic API credits. A typical paper-to-notebook conversion uses ~100k–300k tokens depending on paper length and complexity.
- **Time**: End-to-end generation takes roughly **5–20 minutes** depending on paper complexity.
- **PDF quality**: The pipeline works best with machine-readable PDFs. Scanned-image PDFs may produce poor results.
- **Notebook correctness**: Generated notebooks are best-effort. Always review and test the output before relying on results.
- **Colab target**: Notebooks are designed for Google Colab with a T4 GPU. Adjust hyperparameters for other environments.

---

## 📄 License

This project is provided as-is for personal and educational use.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for:

- Better agent prompts and system instructions
- Support for additional paper formats (arXiv HTML, etc.)
- UI improvements and additional output formats
- Error handling and retry logic
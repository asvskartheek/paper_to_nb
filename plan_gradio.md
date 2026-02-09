# Implementation Plan: Gradio Interface for Paper → Notebook Pipeline

## Overview

Build a **Gradio web interface** that accepts a research paper (PDF), processes it through a two-stage Claude Agent SDK pipeline, and outputs a downloadable Jupyter notebook (`.ipynb`).

**BYO API Key Model**: Users provide their own Anthropic API key in the UI. The key is passed per-session to the Claude Agent SDK via environment variables and is **never stored** on the server. All API costs are borne by the user.

**Pipeline:**
```
API Key + PDF Upload → [paper-to-plan agent] → plan.md → [plan-to-notebook agent] → transformer.ipynb → Download
```

---

## Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        Gradio Interface                          │
│                                                                  │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  🔑 API Key │  │  PDF Upload  │  │  Download .ipynb         │  │
│  │  (Textbox)  │  │  (File)      │  │  (File Component)        │  │
│  └─────┬───────┘  └──────┬───────┘  └──────────────────────────┘  │
│        │                 │                      ▲               │
│        ▼                 ▼                      │               │
│  ┌─────────────────────────────────────────┬─────┴──────────┐   │
│  │         async process_paper()           │                │   │
│  │  api_key ──▶ env={ANTHROPIC_API_KEY}    │                │   │
│  │                                         │                │   │
│  │  Step 1: paper-to-plan (query())        │  Status Logs   │   │
│  │    - Reads PDF via Read tool            │  (Markdown)    │   │
│  │    - Writes plan.md via Write tool      │                │   │
│  │                                         │                │   │
│  │  Step 2: plan-to-notebook (query())     │                │   │
│  │    - Reads plan.md via Read tool        │                │   │
│  │    - Writes .ipynb via Write/Bash tool  │                │   │
│  │    - Returns path to .ipynb             │                │   │
│  └─────────────────────────────────────────┘                │   │
│                                                              │   │
└──────────────────────────────────────────────────────────────────┘
```

> **Key detail**: The user's API key is passed to each `query()` call via
> `ClaudeAgentOptions(env={"ANTHROPIC_API_KEY": api_key})`. It lives only
> in memory for the duration of the request and is never logged or persisted.

### Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| UI | Gradio (Blocks API) | File upload, status display, file download |
| Agent Orchestration | `claude-agent-sdk` (Python) | Two-stage agent pipeline |
| Agent Architecture | `query()` with `AgentDefinition` subagents | Delegating paper-to-plan and plan-to-notebook tasks |
| Async Runtime | `asyncio` / `anyio` | Running async SDK calls within Gradio |
| PDF Handling | Copy to working directory | Claude's `Read` tool can read PDFs natively |

---

## Detailed Design

### 1. Project Structure

```
paper_to_nb/
├── app.py                  # Gradio application (new)
├── main.py                 # Existing entry point
├── plan.md                 # Example plan (existing)
├── plan_gradio.md          # This plan (new)
├── pyproject.toml          # Dependencies
├── README.md               # Project readme
├── transformer.ipynb       # Example output notebook (existing)
└── workspaces/             # Temp working directories (created at runtime)
    └── <uuid>/             # Per-session workspace
        ├── paper.pdf       # Uploaded PDF
        ├── plan.md         # Generated plan
        └── output.ipynb    # Generated notebook
```

### 2. Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "claude-agent-sdk>=0.1.33",
    "gradio>=5.0.0",
]
```

### 3. Gradio UI Design (`app.py`)

#### Layout (using `gr.Blocks`)

```
┌─────────────────────────────────────────────────────────────┐
│                  📄 Paper to Notebook                        │
│         Convert research papers to implementation notebooks  │
│                  🔑 Bring Your Own API Key                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  🔑 Anthropic API Key   (gr.Textbox, type="password")   ││
│  │     [placeholder: "sk-ant-..."]                         ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌─────────────────────────────┐                            │
│  │   Upload PDF               │                            │
│  │   [file_types=[".pdf"]]    │                            │
│  └─────────────────────────────┘                            │
│                                                             │
│  [ 🚀 Generate Notebook ]     (gr.Button, variant="primary")│
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Status / Progress Log (gr.Markdown)                    ││
│  │  - Step 1: Analyzing paper...                           ││
│  │  - Step 2: Generating plan...                           ││
│  │  - Step 3: Building notebook...                         ││
│  │  - ✅ Done!                                             ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  ┌──────────────────────────┐  ┌──────────────────────────┐│
│  │  📋 Plan Preview         │  │  📓 Download Notebook    ││
│  │  (gr.Markdown, plan.md)  │  │  (gr.File, .ipynb)       ││
│  └──────────────────────────┘  └──────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Component Definitions

```python
import gradio as gr

with gr.Blocks(title="Paper to Notebook", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 📄 Paper to Notebook\n"
        "Convert research papers into implementation notebooks\n\n"
        "*🔑 Bring Your Own API Key — you pay only for what you use*"
    )

    # ── API Key Input ──────────────────────────────────────────
    api_key_input = gr.Textbox(
        label="🔑 Anthropic API Key",
        placeholder="sk-ant-api03-...",
        type="password",          # Masked input, never shown in plaintext
        info="Your key is used only for this session and is never stored.",
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload Research Paper (PDF)",
                file_types=[".pdf"],
                file_count="single",
            )
            generate_btn = gr.Button("🚀 Generate Notebook", variant="primary", interactive=False)

        with gr.Column(scale=1):
            status_output = gr.Markdown(label="Status", value="*Enter your API key and upload a PDF to get started...*")

    with gr.Row():
        with gr.Column(scale=1):
            plan_output = gr.Markdown(label="Generated Plan", value="")

        with gr.Column(scale=1):
            notebook_output = gr.File(label="Download Notebook", interactive=False)

    # Enable button only when BOTH api key and PDF are provided
    def check_ready(api_key, pdf_file):
        ready = bool(api_key and api_key.strip().startswith("sk-") and pdf_file is not None)
        return gr.update(interactive=ready)

    api_key_input.change(fn=check_ready, inputs=[api_key_input, pdf_input], outputs=[generate_btn])
    pdf_input.change(fn=check_ready, inputs=[api_key_input, pdf_input], outputs=[generate_btn])

    # Main generation pipeline — api_key is now an input
    generate_btn.click(
        fn=process_paper,
        inputs=[api_key_input, pdf_input],
        outputs=[status_output, plan_output, notebook_output],
    )
```

### 4. Agent Pipeline Design

#### Approach: `query()` with Subagents via `AgentDefinition`

We use the `query()` function (stateless, one-shot) since each pipeline run is independent. We define two subagents programmatically:

1. **`paper-to-plan`**: Reads the PDF, analyzes the paper, and writes a detailed `plan.md`
2. **`plan-to-notebook`**: Reads `plan.md` and generates a complete `.ipynb` notebook

**Why subagents over two separate `query()` calls?**
- Subagents are invoked via the `Task` tool, giving the orchestrator control over sequencing
- The orchestrator agent can pass context between subagents
- Single session manages the full pipeline

**Alternative approach (two sequential `query()` calls):**
- Simpler to implement and debug
- Each step is fully independent
- Easier error handling per step
- **We'll use this approach** for its simplicity and reliability

#### Agent 1: Paper-to-Plan

```python
async def run_paper_to_plan(pdf_path: str, workspace_dir: str, api_key: str) -> str:
    """
    Reads a research paper PDF and generates a detailed implementation plan.
    Returns the path to the generated plan.md file.
    """
    plan_path = os.path.join(workspace_dir, "plan.md")

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are an expert ML researcher and engineer. Your task is to read a research paper "
            "and create a detailed, actionable implementation plan that can be used to build a "
            "complete Jupyter notebook implementing the paper's key ideas.\n\n"
            "The plan should include:\n"
            "- Paper summary\n"
            "- Dataset choice (small enough for a demo, e.g., Google Colab T4 GPU)\n"
            "- Model configuration (scaled down for demo purposes)\n"
            "- Cell-by-cell implementation plan for the notebook\n"
            "- Key paper details that must be faithfully implemented\n"
            "- Training and evaluation strategy\n"
            "- Dependencies and estimated training time\n\n"
            "Write the plan to plan.md in the current working directory."
        ),
        allowed_tools=["Read", "Write", "Glob", "Grep"],
        permission_mode="acceptEdits",
        cwd=workspace_dir,
        max_turns=30,
        env={"ANTHROPIC_API_KEY": api_key},  # BYO API key
    )

    prompt = (
        f"Read the research paper at '{pdf_path}' and create a detailed implementation plan. "
        f"Write the plan to '{plan_path}'. The plan should be thorough enough that another agent "
        f"can use it to generate a complete, runnable Jupyter notebook."
    )

    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result_text += block.text
        elif isinstance(message, ResultMessage):
            if message.is_error:
                raise RuntimeError(f"Paper-to-plan failed: {message.result}")

    return plan_path
```

#### Agent 2: Plan-to-Notebook

```python
async def run_plan_to_notebook(plan_path: str, workspace_dir: str, api_key: str) -> str:
    """
    Reads a plan.md file and generates a complete Jupyter notebook.
    Returns the path to the generated .ipynb file.
    """
    notebook_path = os.path.join(workspace_dir, "output.ipynb")

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are an expert ML engineer. Your task is to read an implementation plan "
            "and generate a complete, runnable Jupyter notebook (.ipynb) that faithfully "
            "implements the plan.\n\n"
            "Requirements:\n"
            "- The notebook must be self-contained and runnable top-to-bottom\n"
            "- Include all necessary imports, installs, and setup cells\n"
            "- Follow the cell structure described in the plan\n"
            "- Include markdown cells with explanations between code cells\n"
            "- Code should be clean, well-commented, and follow best practices\n"
            "- The notebook should be runnable on Google Colab with a T4 GPU\n\n"
            "Write the notebook as a valid .ipynb JSON file."
        ),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit"],
        permission_mode="acceptEdits",
        cwd=workspace_dir,
        max_turns=50,
        env={"ANTHROPIC_API_KEY": api_key},  # BYO API key
    )

    prompt = (
        f"Read the implementation plan at '{plan_path}' and generate a complete Jupyter notebook. "
        f"Write the notebook to '{notebook_path}'. The notebook should be immediately runnable "
        f"and faithfully implement everything described in the plan."
    )

    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result_text += block.text
        elif isinstance(message, ResultMessage):
            if message.is_error:
                raise RuntimeError(f"Plan-to-notebook failed: {message.result}")

    return notebook_path
```

### 5. Orchestrator Function (Gradio Handler)

```python
import os
import uuid
import shutil
import asyncio
from pathlib import Path

WORKSPACES_DIR = Path(__file__).parent / "workspaces"

async def _process_paper_async(api_key: str, pdf_file) -> tuple[str, str, str | None]:
    """
    Main pipeline: PDF → plan.md → notebook.ipynb
    Returns: (status_markdown, plan_markdown, notebook_path_or_none)
    """
    # 1. Create isolated workspace
    workspace_id = str(uuid.uuid4())[:8]
    workspace_dir = WORKSPACES_DIR / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)

    status_lines = []

    try:
        # 2. Copy PDF to workspace
        pdf_dest = workspace_dir / "paper.pdf"
        shutil.copy2(pdf_file, pdf_dest)
        status_lines.append("✅ **Step 1/3**: PDF uploaded and ready")

        # 3. Run paper-to-plan agent (api_key passed per-call)
        status_lines.append("⏳ **Step 2/3**: Analyzing paper and generating plan...")
        plan_path = await run_paper_to_plan(str(pdf_dest), str(workspace_dir), api_key)

        if not os.path.exists(plan_path):
            raise FileNotFoundError("Plan file was not generated")

        plan_content = Path(plan_path).read_text()
        status_lines[-1] = "✅ **Step 2/3**: Plan generated successfully"

        # 4. Run plan-to-notebook agent (api_key passed per-call)
        status_lines.append("⏳ **Step 3/3**: Generating notebook from plan...")
        notebook_path = await run_plan_to_notebook(plan_path, str(workspace_dir), api_key)

        if not os.path.exists(notebook_path):
            raise FileNotFoundError("Notebook file was not generated")

        status_lines[-1] = "✅ **Step 3/3**: Notebook generated successfully"
        status_lines.append("\n🎉 **Done!** Download your notebook below.")

        return "\n\n".join(status_lines), plan_content, notebook_path

    except Exception as e:
        status_lines.append(f"\n❌ **Error**: {str(e)}")
        return "\n\n".join(status_lines), "", None


def process_paper(api_key: str, pdf_file) -> tuple[str, str, str | None]:
    """Sync wrapper for the async pipeline (Gradio handler)."""
    if not api_key or not api_key.strip():
        return "⚠️ Please enter your Anthropic API key.", "", None
    if not api_key.strip().startswith("sk-"):
        return "⚠️ Invalid API key format. It should start with `sk-`.", "", None
    if pdf_file is None:
        return "⚠️ Please upload a PDF file first.", "", None

    # Run the async pipeline — api_key is passed through, never stored
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_process_paper_async(api_key.strip(), pdf_file))
    finally:
        loop.close()
```

### 6. Streaming Status Updates (Enhanced UX)

For a better user experience, use **Gradio's generator pattern** to yield incremental status updates:

```python
def process_paper_streaming(api_key: str, pdf_file):
    """
    Generator function that yields incremental status updates.
    Gradio will update the UI after each yield.
    """
    if not api_key or not api_key.strip().startswith("sk-"):
        yield "⚠️ Please enter a valid Anthropic API key (starts with `sk-`).", "", None
        return
    if pdf_file is None:
        yield "⚠️ Please upload a PDF file first.", "", None
        return

    api_key = api_key.strip()
    workspace_id = str(uuid.uuid4())[:8]
    workspace_dir = WORKSPACES_DIR / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy PDF
    pdf_dest = workspace_dir / "paper.pdf"
    shutil.copy2(pdf_file, pdf_dest)
    yield "✅ PDF uploaded\n\n⏳ **Analyzing paper and generating plan...**\n\n*(This may take 2-5 minutes)*", "", None

    # Step 2: Paper to Plan
    loop = asyncio.new_event_loop()
    try:
        plan_path = loop.run_until_complete(
            run_paper_to_plan(str(pdf_dest), str(workspace_dir), api_key)
        )
    except Exception as e:
        yield f"❌ **Error in plan generation**: {e}", "", None
        return
    finally:
        loop.close()

    plan_content = Path(plan_path).read_text() if os.path.exists(plan_path) else ""
    yield (
        "✅ Plan generated\n\n⏳ **Generating notebook...**\n\n*(This may take 5-15 minutes)*",
        plan_content,
        None,
    )

    # Step 3: Plan to Notebook
    loop = asyncio.new_event_loop()
    try:
        notebook_path = loop.run_until_complete(
            run_plan_to_notebook(plan_path, str(workspace_dir), api_key)
        )
    except Exception as e:
        yield f"✅ Plan generated\n\n❌ **Error in notebook generation**: {e}", plan_content, None
        return
    finally:
        loop.close()

    if os.path.exists(notebook_path):
        yield (
            "✅ Plan generated\n\n✅ Notebook generated\n\n🎉 **Done! Download your notebook below.**",
            plan_content,
            notebook_path,
        )
    else:
        yield (
            "✅ Plan generated\n\n❌ **Notebook file was not created**",
            plan_content,
            None,
        )
```

**Wire up the generator to Gradio:**
```python
generate_btn.click(
    fn=process_paper_streaming,
    inputs=[api_key_input, pdf_input],
    outputs=[status_output, plan_output, notebook_output],
)
```

### 7. Alternative: Using Subagents (Single `query()` Call)

If you prefer a single orchestrated `query()` call with subagents:

```python
async def run_pipeline_with_subagents(pdf_path: str, workspace_dir: str, api_key: str):
    """Use AgentDefinition subagents for a single-session pipeline."""

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit", "Task"],
        permission_mode="acceptEdits",
        cwd=workspace_dir,
        max_turns=80,
        env={"ANTHROPIC_API_KEY": api_key},  # BYO API key
        agents={
            "paper-to-plan": AgentDefinition(
                description=(
                    "Expert at reading research papers and creating detailed implementation plans. "
                    "Use this agent when you need to analyze a PDF research paper and produce a "
                    "structured plan.md file."
                ),
                prompt=(
                    "You are an expert ML researcher. Read the research paper provided and create "
                    "a detailed implementation plan in plan.md. Include: paper summary, dataset choice "
                    "(small enough for Colab T4), model config (scaled down), cell-by-cell notebook plan, "
                    "key paper details, training/eval strategy, dependencies, and estimated times."
                ),
                tools=["Read", "Write", "Glob", "Grep"],
                model="sonnet",
            ),
            "plan-to-notebook": AgentDefinition(
                description=(
                    "Expert at converting implementation plans into complete, runnable Jupyter notebooks. "
                    "Use this agent when you have a plan.md and need to generate a .ipynb file."
                ),
                prompt=(
                    "You are an expert ML engineer. Read the plan.md file and generate a complete, "
                    "self-contained Jupyter notebook (.ipynb). The notebook must be runnable top-to-bottom "
                    "on Google Colab with a T4 GPU. Include all imports, installs, model code, training, "
                    "and evaluation cells with markdown explanations."
                ),
                tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit"],
                model="sonnet",
            ),
        },
    )

    prompt = (
        f"I need you to orchestrate a two-step pipeline:\n\n"
        f"1. First, use the 'paper-to-plan' agent to read the research paper at "
        f"'{pdf_path}' and create a plan.md file in the current directory.\n\n"
        f"2. Then, use the 'plan-to-notebook' agent to read the plan.md and generate "
        f"a complete output.ipynb notebook file.\n\n"
        f"Execute both steps in order. Confirm when complete."
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            if message.is_error:
                raise RuntimeError(f"Pipeline failed: {message.result}")
            return message
```

**Trade-offs:**

| Aspect | Two `query()` calls | Single `query()` + subagents |
|---|---|---|
| Simplicity | ✅ Simpler, easier to debug | More complex orchestration |
| Error handling | ✅ Per-step error handling | Harder to isolate failures |
| Progress updates | ✅ Natural step boundaries | Need to parse messages |
| Context sharing | Manual (file system) | ✅ Automatic via orchestrator |
| Token efficiency | Separate sessions | ✅ Shared context |
| Reliability | ✅ Independent retries | All-or-nothing |

**Recommendation**: Start with **two sequential `query()` calls** for simplicity and reliability. Migrate to subagents later if context-sharing becomes important.

---

## Implementation Steps

### Step 1: Update Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "claude-agent-sdk>=0.1.33",
    "gradio>=5.0.0",
]
```

### Step 2: Create `app.py`

Full implementation file combining:
- Gradio Blocks UI
- `run_paper_to_plan()` async function
- `run_plan_to_notebook()` async function
- `process_paper_streaming()` generator
- `demo.launch()` entry point

### Step 3: Run & Test

```bash
# Install deps
pip install -e .
# or
pip install claude-agent-sdk gradio

# Run the app
python app.py
# Opens at http://localhost:7860
```

---

## Error Handling Strategy

| Error | Cause | Handling |
|---|---|---|
| Missing / invalid API key | User didn't enter key or key format wrong | Validate before calling agents; show clear message |
| `CLINotFoundError` | Claude Code CLI not installed | Show install instructions in status |
| `ProcessError` (exit 401) | Invalid or expired API key | Show "Invalid API key. Check your key at console.anthropic.com" |
| `ProcessError` (exit 429) | Rate limit / insufficient credits | Show "Rate limited. Check your Anthropic usage/billing" |
| `ProcessError` | Agent process crashed | Show error details, suggest retry |
| `FileNotFoundError` | Agent didn't create expected file | Show what was expected vs. found |
| `TimeoutError` | Agent took too long | Set `max_turns` limit, show timeout |
| Workspace errors | Disk space, permissions | Catch `OSError`, show details |

```python
from claude_agent_sdk import CLINotFoundError, ProcessError, CLIJSONDecodeError

try:
    plan_path = await run_paper_to_plan(pdf_path, workspace_dir)
except CLINotFoundError:
    return "❌ Claude Code CLI not found. Install with: `npm install -g @anthropic-ai/claude-code`", "", None
except ProcessError as e:
    return f"❌ Agent process failed (exit code {e.exit_code}): {e.stderr}", "", None
except Exception as e:
    return f"❌ Unexpected error: {str(e)}", "", None
```

---

## Configuration & Environment

### BYO API Key — No Server-Side Auth Required

The app does **not** need any pre-configured API key or `claude login` on the server.
Each user provides their own Anthropic API key in the UI, which is passed directly
to the Claude Agent SDK via `ClaudeAgentOptions(env={"ANTHROPIC_API_KEY": key})`.

**How it works:**
1. User pastes their API key into the masked `gr.Textbox(type="password")` field
2. On clicking "Generate", the key is passed in-memory to the agent functions
3. Each `query()` call receives the key via `env=` — it is set only for that subprocess
4. After the request completes, the key is garbage-collected — **never stored to disk**

**Where to get an API key:**
- Go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- Create a new API key
- Paste it into the app

### Server Requirements

```bash
# Claude Code CLI must be installed on the server (but NOT authenticated):
npm install -g @anthropic-ai/claude-code
# or it's bundled with: pip install claude-agent-sdk
```

### Optional Configuration

```python
# In app.py - configurable constants
MAX_PDF_SIZE_MB = 50
MAX_PLAN_TURNS = 30        # Max agent turns for plan generation
MAX_NOTEBOOK_TURNS = 50    # Max agent turns for notebook generation
WORKSPACE_CLEANUP = True   # Auto-delete workspaces after download
```

---

## Security Considerations

1. **API Key Handling (critical)**:
   - Keys entered via `gr.Textbox(type="password")` — masked in the UI
   - Passed only via `ClaudeAgentOptions(env=...)` — scoped to the subprocess
   - **Never logged**, written to disk, or stored in any database
   - Garbage-collected after the request completes
   - If deploying publicly, **use HTTPS** to encrypt keys in transit
   - Consider adding `gr.Markdown` disclaimer about key safety
2. **Isolated Workspaces**: Each upload gets a unique workspace directory via UUID
3. **Permission Mode**: Use `acceptEdits` (not `bypassPermissions`) to auto-approve file operations while maintaining safety
4. **Tool Restrictions**: 
   - Paper-to-plan: Read-only + Write (no Bash)
   - Plan-to-notebook: Read + Write + Edit + Bash + NotebookEdit (needs Bash for pip installs in notebook testing)
5. **File Size Limits**: Enforce max upload size via Gradio's `max_file_size` parameter
6. **Workspace Cleanup**: Periodically clean up old workspace directories
7. **No API key caching**: Intentionally do NOT offer "remember my key" to avoid storage risks

---

## Testing Plan

1. **Unit Tests**:
   - Test workspace creation and cleanup
   - Test PDF copy to workspace
   - Test plan file parsing

2. **Integration Tests**:
   - Upload a known paper (e.g., "Attention Is All You Need")
   - Verify plan.md is generated with expected sections
   - Verify .ipynb is valid JSON and has expected cell structure

3. **Manual Testing**:
   - Upload various paper PDFs (different sizes, formats)
   - Verify Gradio UI responsiveness during long agent runs
   - Test error cases (no PDF, corrupt PDF, network issues)

---

## Future Enhancements

1. **Real-time streaming**: Use `ClaudeSDKClient` with `receive_messages()` to stream agent thinking/progress to the UI in real-time
2. **Plan editing**: Let users edit the plan.md in the UI before generating the notebook
3. **Model selection**: Dropdown to choose between sonnet/opus/haiku for agents
4. **Multiple papers**: Support batch processing of multiple PDFs
5. **Template selection**: Pre-built plan templates for common paper types (NLP, CV, RL)
6. **Notebook validation**: Auto-run the notebook in a sandbox to verify it executes correctly
7. **Cost tracking**: Display token usage and cost from `ResultMessage.total_cost_usd`
8. **Session persistence**: Save and resume pipeline sessions using `resume` option

---

## Complete `app.py` Skeleton

```python
"""
Paper to Notebook - Gradio Application (BYO API Key)
Converts research papers (PDF) to implementation notebooks (.ipynb)
using Claude Agent SDK. Users provide their own Anthropic API key.
"""

import os
import uuid
import shutil
import asyncio
from pathlib import Path

import gradio as gr
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ResultMessage,
    CLINotFoundError,
    ProcessError,
)

# ── Constants ──────────────────────────────────────────────────
WORKSPACES_DIR = Path(__file__).parent / "workspaces"
MAX_PLAN_TURNS = 30
MAX_NOTEBOOK_TURNS = 50

# ── Agent Functions ────────────────────────────────────────────

async def run_paper_to_plan(pdf_path: str, workspace_dir: str, api_key: str) -> str:
    """Agent 1: Read paper PDF → write plan.md"""
    plan_path = os.path.join(workspace_dir, "plan.md")
    options = ClaudeAgentOptions(
        system_prompt="... (detailed system prompt for plan generation) ...",
        allowed_tools=["Read", "Write", "Glob", "Grep"],
        permission_mode="acceptEdits",
        cwd=workspace_dir,
        max_turns=MAX_PLAN_TURNS,
        env={"ANTHROPIC_API_KEY": api_key},  # BYO key
    )
    async for message in query(
        prompt=f"Read the research paper at '{pdf_path}' and create a detailed plan at '{plan_path}'.",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.is_error:
            raise RuntimeError(f"Plan generation failed: {message.result}")
    return plan_path


async def run_plan_to_notebook(plan_path: str, workspace_dir: str, api_key: str) -> str:
    """Agent 2: Read plan.md → write output.ipynb"""
    notebook_path = os.path.join(workspace_dir, "output.ipynb")
    options = ClaudeAgentOptions(
        system_prompt="... (detailed system prompt for notebook generation) ...",
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit"],
        permission_mode="acceptEdits",
        cwd=workspace_dir,
        max_turns=MAX_NOTEBOOK_TURNS,
        env={"ANTHROPIC_API_KEY": api_key},  # BYO key
    )
    async for message in query(
        prompt=f"Read the plan at '{plan_path}' and generate a complete notebook at '{notebook_path}'.",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.is_error:
            raise RuntimeError(f"Notebook generation failed: {message.result}")
    return notebook_path


# ── Gradio Handler (Generator for streaming updates) ──────────

def process_paper(api_key: str, pdf_file):
    """Main pipeline handler - yields incremental status updates."""
    # ── Validate inputs ──
    if not api_key or not api_key.strip().startswith("sk-"):
        yield "⚠️ Enter a valid Anthropic API key (starts with `sk-`).", "", None
        return
    if pdf_file is None:
        yield "⚠️ Upload a PDF first.", "", None
        return

    api_key = api_key.strip()  # use only in-memory, never persist

    workspace_dir = WORKSPACES_DIR / str(uuid.uuid4())[:8]
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Copy PDF
    pdf_dest = workspace_dir / "paper.pdf"
    shutil.copy2(pdf_file, pdf_dest)
    yield "✅ PDF uploaded\n\n⏳ **Generating plan...** *(2-5 min)*", "", None

    # Step 1: Plan
    loop = asyncio.new_event_loop()
    try:
        plan_path = loop.run_until_complete(
            run_paper_to_plan(str(pdf_dest), str(workspace_dir), api_key)
        )
    except Exception as e:
        yield f"❌ Plan generation failed: {e}", "", None
        return
    finally:
        loop.close()

    plan_text = Path(plan_path).read_text() if os.path.exists(plan_path) else ""
    yield "✅ Plan generated\n\n⏳ **Generating notebook...** *(5-15 min)*", plan_text, None

    # Step 2: Notebook
    loop = asyncio.new_event_loop()
    try:
        nb_path = loop.run_until_complete(
            run_plan_to_notebook(plan_path, str(workspace_dir), api_key)
        )
    except Exception as e:
        yield f"✅ Plan generated\n\n❌ Notebook generation failed: {e}", plan_text, None
        return
    finally:
        loop.close()

    if os.path.exists(nb_path):
        yield "✅ Plan generated\n\n✅ Notebook generated\n\n🎉 **Done!**", plan_text, nb_path
    else:
        yield "✅ Plan generated\n\n❌ Notebook file not found", plan_text, None


# ── Gradio UI ──────────────────────────────────────────────────

with gr.Blocks(title="Paper to Notebook", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 📄 Paper to Notebook\n"
        "Convert research papers into implementation notebooks\n\n"
        "*🔑 Bring Your Own API Key — you pay only for what you use*"
    )

    api_key_input = gr.Textbox(
        label="🔑 Anthropic API Key",
        placeholder="sk-ant-api03-...",
        type="password",
        info="Your key is used only for this session and is never stored.",
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload Research Paper (PDF)", file_types=[".pdf"])
            generate_btn = gr.Button("🚀 Generate Notebook", variant="primary", interactive=False)
        with gr.Column(scale=1):
            status_output = gr.Markdown(value="*Enter your API key and upload a PDF to get started...*")

    with gr.Row():
        with gr.Column(scale=1):
            plan_output = gr.Markdown(label="Generated Plan", value="")
        with gr.Column(scale=1):
            notebook_output = gr.File(label="Download Notebook", interactive=False)

    # Enable button only when BOTH api key and PDF are provided
    def check_ready(api_key, pdf_file):
        ready = bool(api_key and api_key.strip().startswith("sk-") and pdf_file is not None)
        return gr.update(interactive=ready)

    api_key_input.change(fn=check_ready, inputs=[api_key_input, pdf_input], outputs=[generate_btn])
    pdf_input.change(fn=check_ready, inputs=[api_key_input, pdf_input], outputs=[generate_btn])

    generate_btn.click(
        fn=process_paper,
        inputs=[api_key_input, pdf_input],
        outputs=[status_output, plan_output, notebook_output],
    )

if __name__ == "__main__":
    demo.launch()
```

---

## Summary

| Aspect | Decision |
|---|---|
| **API key model** | **BYO — user provides their own Anthropic API key in the UI** |
| **Key handling** | **Passed via `env={"ANTHROPIC_API_KEY": key}`, never stored** |
| SDK function | `query()` (stateless, two sequential calls) |
| Agent architecture | Two independent agents, not subagents |
| Gradio pattern | `gr.Blocks` with generator for streaming updates |
| File handling | Isolated workspace per session (UUID-based) |
| PDF reading | Claude's native `Read` tool (supports PDFs) |
| Notebook writing | Claude's `Write` tool (writes .ipynb JSON) |
| Permission mode | `acceptEdits` (auto-approve file ops) |
| Error handling | Per-step try/catch with user-friendly messages |
| Progress | Generator yields after each step |
| Cost model | User bears all API costs via their own key |

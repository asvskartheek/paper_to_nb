"""
Paper to Notebook - Gradio Application (BYO API Key)

Converts research papers (PDF) to implementation notebooks (.ipynb)
using Claude Agent SDK. Users provide their own Anthropic API key.

Pipeline:
    API Key + PDF Upload → [paper-to-plan agent] → plan.md
                         → [plan-to-notebook agent] → output.ipynb → Download
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

# ── Constants ──────────────────────────────────────────────────────────────────
WORKSPACES_DIR = Path(__file__).parent / "workspaces"
MAX_PDF_SIZE_MB = 50
MAX_PLAN_TURNS = 30
MAX_NOTEBOOK_TURNS = 50


# ── Agent Functions ────────────────────────────────────────────────────────────

async def run_paper_to_plan(pdf_path: str, workspace_dir: str, api_key: str) -> str:
    """
    Agent 1: Read a research paper PDF and generate a detailed implementation plan.

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
        max_turns=MAX_PLAN_TURNS,
        env={"ANTHROPIC_API_KEY": api_key},
    )

    prompt = (
        f"Read the research paper at '{pdf_path}' and create a detailed implementation plan. "
        f"Write the plan to '{plan_path}'. The plan should be thorough enough that another agent "
        f"can use it to generate a complete, runnable Jupyter notebook."
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage) and message.is_error:
            raise RuntimeError(f"Plan generation failed: {message.result}")

    return plan_path


async def run_plan_to_notebook(plan_path: str, workspace_dir: str, api_key: str) -> str:
    """
    Agent 2: Read plan.md and generate a complete Jupyter notebook.

    Returns the path to the generated output.ipynb file.
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
        max_turns=MAX_NOTEBOOK_TURNS,
        env={"ANTHROPIC_API_KEY": api_key},
    )

    prompt = (
        f"Read the implementation plan at '{plan_path}' and generate a complete Jupyter notebook. "
        f"Write the notebook to '{notebook_path}'. The notebook should be immediately runnable "
        f"and faithfully implement everything described in the plan."
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage) and message.is_error:
            raise RuntimeError(f"Notebook generation failed: {message.result}")

    return notebook_path


# ── Gradio Handler (Generator for streaming updates) ──────────────────────────

def process_paper(api_key: str, pdf_file):
    """
    Main pipeline handler — yields incremental status updates.

    The API key is used only in-memory for this request and is never
    logged, stored, or persisted to disk.
    """
    # ── Validate inputs ──
    if not api_key or not api_key.strip():
        yield "⚠️ Please enter your Anthropic API key.", "", None
        return
    if not api_key.strip().startswith("sk-"):
        yield "⚠️ Invalid API key format. It should start with `sk-`.", "", None
        return
    if pdf_file is None:
        yield "⚠️ Please upload a PDF file first.", "", None
        return

    api_key = api_key.strip()  # use only in-memory, never persist

    # ── Create isolated workspace ──
    workspace_id = str(uuid.uuid4())[:8]
    workspace_dir = WORKSPACES_DIR / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Copy PDF to workspace ──
    pdf_dest = workspace_dir / "paper.pdf"
    shutil.copy2(pdf_file, pdf_dest)
    yield (
        "✅ PDF uploaded\n\n⏳ **Analyzing paper and generating plan...**\n\n"
        "*(This may take 2-5 minutes)*",
        "",
        None,
    )

    # ── Step 2: Paper → Plan ──
    loop = asyncio.new_event_loop()
    try:
        plan_path = loop.run_until_complete(
            run_paper_to_plan(str(pdf_dest), str(workspace_dir), api_key)
        )
    except CLINotFoundError:
        yield (
            "❌ Claude Code CLI not found. "
            "Install with: `npm install -g @anthropic-ai/claude-code`",
            "",
            None,
        )
        return
    except ProcessError as e:
        if "401" in str(e):
            yield (
                "❌ **Invalid or expired API key.** "
                "Check your key at [console.anthropic.com](https://console.anthropic.com/settings/keys).",
                "",
                None,
            )
        elif "429" in str(e):
            yield (
                "❌ **Rate limited.** Check your Anthropic usage and billing.",
                "",
                None,
            )
        else:
            yield f"❌ **Plan generation failed**: {e}", "", None
        return
    except Exception as e:
        yield f"❌ **Plan generation failed**: {e}", "", None
        return
    finally:
        loop.close()

    # Read the generated plan
    plan_text = ""
    if os.path.exists(plan_path):
        plan_text = Path(plan_path).read_text()
    else:
        yield (
            "✅ PDF uploaded\n\n❌ **Plan file was not generated.** "
            "The agent may not have created plan.md.",
            "",
            None,
        )
        return

    yield (
        "✅ Plan generated\n\n⏳ **Generating notebook from plan...**\n\n"
        "*(This may take 5-15 minutes)*",
        plan_text,
        None,
    )

    # ── Step 3: Plan → Notebook ──
    loop = asyncio.new_event_loop()
    try:
        notebook_path = loop.run_until_complete(
            run_plan_to_notebook(plan_path, str(workspace_dir), api_key)
        )
    except CLINotFoundError:
        yield (
            "✅ Plan generated\n\n"
            "❌ Claude Code CLI not found. "
            "Install with: `npm install -g @anthropic-ai/claude-code`",
            plan_text,
            None,
        )
        return
    except ProcessError as e:
        if "401" in str(e):
            yield (
                "✅ Plan generated\n\n"
                "❌ **Invalid or expired API key.** "
                "Check your key at [console.anthropic.com](https://console.anthropic.com/settings/keys).",
                plan_text,
                None,
            )
        elif "429" in str(e):
            yield (
                "✅ Plan generated\n\n"
                "❌ **Rate limited.** Check your Anthropic usage and billing.",
                plan_text,
                None,
            )
        else:
            yield f"✅ Plan generated\n\n❌ **Notebook generation failed**: {e}", plan_text, None
        return
    except Exception as e:
        yield f"✅ Plan generated\n\n❌ **Notebook generation failed**: {e}", plan_text, None
        return
    finally:
        loop.close()

    # ── Final result ──
    if os.path.exists(notebook_path):
        yield (
            "✅ Plan generated\n\n✅ Notebook generated\n\n"
            "🎉 **Done! Download your notebook below.**",
            plan_text,
            notebook_path,
        )
    else:
        yield (
            "✅ Plan generated\n\n❌ **Notebook file was not created.**",
            plan_text,
            None,
        )


# ── UI Helper ─────────────────────────────────────────────────────────────────

def check_ready(api_key, pdf_file):
    """Enable the Generate button only when both API key and PDF are provided."""
    ready = bool(
        api_key
        and api_key.strip().startswith("sk-")
        and pdf_file is not None
    )
    return gr.update(interactive=ready)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Paper to Notebook") as demo:
    gr.Markdown(
        "# 📄 Paper to Notebook\n"
        "Convert research papers into implementation notebooks\n\n"
        "*🔑 Bring Your Own API Key — you pay only for what you use*"
    )

    # ── API Key Input ──
    api_key_input = gr.Textbox(
        label="🔑 Anthropic API Key",
        placeholder="sk-ant-api03-...",
        type="password",
        info="Your key is used only for this session and is never stored.",
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload Research Paper (PDF)",
                file_types=[".pdf"],
                file_count="single",
            )
            generate_btn = gr.Button(
                "🚀 Generate Notebook",
                variant="primary",
                interactive=False,
            )

        with gr.Column(scale=1):
            status_output = gr.Markdown(
                value="*Enter your API key and upload a PDF to get started...*"
            )

    with gr.Row():
        with gr.Column(scale=1):
            plan_output = gr.Markdown(label="Generated Plan", value="")

        with gr.Column(scale=1):
            notebook_output = gr.File(
                label="Download Notebook",
                interactive=False,
            )

    # ── Event Wiring ──

    # Enable button only when BOTH api key and PDF are provided
    api_key_input.change(
        fn=check_ready,
        inputs=[api_key_input, pdf_input],
        outputs=[generate_btn],
    )
    pdf_input.change(
        fn=check_ready,
        inputs=[api_key_input, pdf_input],
        outputs=[generate_btn],
    )

    # Main generation pipeline
    generate_btn.click(
        fn=process_paper,
        inputs=[api_key_input, pdf_input],
        outputs=[status_output, plan_output, notebook_output],
    )

# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()

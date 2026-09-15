"""English-only CLI adapter for the durable LangGraph RFQ workflow."""

from __future__ import annotations

import shutil
import sqlite3
import traceback
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rfq_agent.chains.model import build_chat_model
from rfq_agent.graph.runtime import GraphExecutionError, RFQGraphRuntime, build_runtime
from rfq_agent.persistence.repository import SQLiteCaseRepository
from rfq_agent.settings import Settings

app = typer.Typer(
    help="Create, review, resume, and export IT RFQs.", pretty_exceptions_enable=False
)
config_app = typer.Typer(help="Check model configuration.")
app.add_typer(config_app, name="config")
console = Console(markup=False)
_debug = False


class SaveAndExit(Exception):
    """The operator chose to leave the current durable interrupt pending."""


def guarded[**P, R](function: Callable[P, R]) -> Callable[P, R | None]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R | None:
        try:
            return function(*args, **kwargs)
        except (SaveAndExit, KeyboardInterrupt, EOFError, typer.Abort):
            console.print("Session stopped. Resume the case from its saved checkpoint.")
            return None
        except (ValueError, KeyError, OSError, sqlite3.Error, GraphExecutionError) as error:
            if _debug:
                traceback.print_exc()
            console.print(f"Error: {error}")
            raise typer.Exit(1) from None

    return wrapped


@app.callback()
def main(
    debug: bool = typer.Option(False, "--debug", help="Show diagnostic stack traces."),
) -> None:
    global _debug
    _debug = debug


def ask(prompt: str) -> str:
    answer = str(typer.prompt(prompt))
    if answer.strip().casefold() in ("quit", "exit"):
        raise SaveAndExit()
    return answer


@config_app.command("check")
@guarded
def config_check() -> None:
    settings = Settings()
    if not settings.llm_model.strip():
        raise ValueError("Set LLM_MODEL to an available model name.")
    if not settings.llm_api_key.get_secret_value():
        raise ValueError("Set LLM_API_KEY in your local environment or .env file.")
    try:
        model = build_chat_model(settings)
        model.invoke("Reply with OK.")
    except Exception as error:
        raise GraphExecutionError(
            f"Model connectivity check failed ({type(error).__name__})."
        ) from error
    console.print("Configuration is valid.")


@app.command("list")
@guarded
def list_cases() -> None:
    repository = SQLiteCaseRepository(Settings().rfq_database_path)
    table = Table("Case ID", "Status", "Request", "Document")
    graph_records = repository.list_graph_cases()
    for record in graph_records:
        table.add_row(
            record.case_id,
            record.status,
            record.initial_request,
            record.document_path or "-",
        )
    graph_ids = {record.case_id for record in graph_records}
    for state in repository.list_legacy_cases():
        if state.case_id not in graph_ids:
            table.add_row(
                state.case_id,
                f"legacy:{state.phase}",
                "Restart required to continue this legacy case.",
                state.document_path or "-",
            )
    console.print(table)


@app.command("new")
@guarded
def new_case() -> None:
    settings = Settings()
    request = ask("Describe the IT procurement need (or quit)")
    case_id = str(uuid4())
    console.print(f"Case ID: {case_id}")
    with build_runtime(settings) as runtime:
        result = runtime.start(case_id, request)
        _run_interrupt_loop(runtime, case_id, result)


@app.command("resume")
@guarded
def resume_case(case_id: str) -> None:
    settings = Settings()
    repository = SQLiteCaseRepository(settings.rfq_database_path)
    try:
        record = repository.graph_case(case_id)
    except KeyError as graph_error:
        try:
            repository.legacy_case(case_id)
        except KeyError:
            raise graph_error
        raise ValueError("Active legacy cases cannot resume; start a new case to restart.")
    if record.status == "completed":
        if not record.document_path:
            raise ValueError("Completed case has no indexed document.")
        console.print(f"Completed. Document: {record.document_path}")
        return
    with build_runtime(settings) as runtime:
        result = runtime.continue_case(case_id)
        _run_interrupt_loop(runtime, case_id, result)


@app.command("export")
@guarded
def export_case(
    case_id: str, output_dir: Path = typer.Option(Path("generated"), "--output-dir")
) -> None:
    repository = SQLiteCaseRepository(Settings().rfq_database_path)
    try:
        source = repository.graph_document(case_id)
    except ValueError as graph_error:
        try:
            legacy = repository.legacy_case(case_id)
        except KeyError:
            raise graph_error
        if legacy.phase != "completed" or not legacy.document_path:
            raise ValueError("Active legacy cases must be restarted before export.")
        source = Path(legacy.document_path)
    if not source.is_file():
        raise OSError(f"Approved document is missing: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    console.print(f"Exported: {destination.resolve()}")


def _run_interrupt_loop(runtime: RFQGraphRuntime, case_id: str, result: dict[str, object]) -> None:
    while result.get("status") != "completed":
        payload = runtime.interrupt_payload(result)
        if payload is None:
            raise GraphExecutionError("The workflow stopped without an input request.")
        prompt = _render_interrupt(payload)
        response = ask(prompt)
        runtime.repository.save_graph_exchange(case_id, assistant=prompt, user=response)
        result = runtime.resume(case_id, response)
    path = str(result["document_path"])
    console.print(f"Generated: {Path(path).resolve()}")


def _render_interrupt(payload: dict[str, object]) -> str:
    kind = payload.get("kind")
    clarification = payload.get("clarification_message")
    if clarification and kind != "question":
        console.print(f"Please clarify: {clarification}")
    if kind == "type_confirmation":
        proposed = str(payload["proposed_type"])
        console.print(Panel(proposed, title="Proposed RFQ type"))
        return "Confirm this RFQ type, or enter the correct type"
    if kind == "question":
        if clarification:
            console.print(f"Question: {payload['wording']}")
        return f"{clarification or payload['wording']} (or skip)"
    if kind == "revision":
        issues = payload.get("issues")
        if isinstance(issues, list):
            for issue in issues:
                if isinstance(issue, dict):
                    console.print(f"Review issue: {issue.get('description', '')}")
        return "Approve this draft, or describe the revision required"
    if kind == "final_approval":
        draft = payload.get("draft")
        if isinstance(draft, dict):
            for section in draft.get("sections", []):
                if isinstance(section, dict):
                    console.print(
                        Panel(
                            str(section.get("body_markdown", "")),
                            title=str(section.get("heading", "")),
                        )
                    )
        return "Approve this draft, or enter a revision request"
    raise GraphExecutionError(f"Unsupported workflow interrupt: {kind}")

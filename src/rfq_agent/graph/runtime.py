"""Durable execution boundary for the LangGraph RFQ workflow."""

from __future__ import annotations

import sqlite3
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast
from uuid import uuid4

from langchain_core.exceptions import OutputParserException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from rfq_agent.chains.factory import build_rfq_chains
from rfq_agent.chains.model import build_chat_model
from rfq_agent.documents.generator import GraphDocumentRenderer
from rfq_agent.graph.builder import build_rfq_graph
from rfq_agent.graph.nodes import DocumentRenderer, RFQChains
from rfq_agent.persistence.repository import SQLiteCaseRepository
from rfq_agent.settings import Settings


class GraphExecutionError(RuntimeError):
    """A graph node failed after LangGraph saved the latest durable state."""


_active_case_id: ContextVar[str | None] = ContextVar("rfq_active_case_id", default=None)


class _AuditedChain:
    def __init__(
        self,
        chain: Any,
        repository: SQLiteCaseRepository,
        *,
        capability: str,
        model: str,
    ) -> None:
        self._chain = chain
        self._repository = repository
        self._capability = capability
        self._model = model

    def invoke(self, payload: dict[str, Any], config: Any = None, **kwargs: Any) -> Any:
        case_id = _active_case_id.get()
        if case_id is None:
            raise RuntimeError("Model invocation is outside an RFQ graph execution.")
        run_id = str(uuid4())
        self._repository.begin_graph_model_run(
            run_id=run_id,
            case_id=case_id,
            capability=self._capability,
            model=self._model,
            started_at=datetime.now(UTC).isoformat(),
        )
        try:
            if config is None:
                result = self._chain.invoke(payload, **kwargs)
            else:
                result = self._chain.invoke(payload, config=config, **kwargs)
        except Exception as error:
            try:
                self._repository.finish_graph_model_run(
                    run_id,
                    status="failed",
                    finished_at=datetime.now(UTC).isoformat(),
                    error_type=type(error).__name__,
                )
            except Exception:
                pass
            raise
        try:
            self._repository.finish_graph_model_run(
                run_id,
                status="succeeded",
                finished_at=datetime.now(UTC).isoformat(),
            )
        except Exception:
            # The provider call already succeeded. Keeping the row as "running"
            # exposes incomplete audit finalization without causing a paid retry.
            pass
        return result


def _with_audit(chains: RFQChains, repository: SQLiteCaseRepository, model_name: str) -> RFQChains:
    audited = type("AuditedRFQChains", (), {})()
    for capability in (
        "classifier",
        "fact_ledger_initializer",
        "planner",
        "answer_validator",
        "type_interpreter",
        "review_interpreter",
        "revision_interpreter",
        "writer",
        "reviewer",
    ):
        setattr(
            audited,
            capability,
            _AuditedChain(
                getattr(chains, capability),
                repository,
                capability=capability,
                model=model_name,
            ),
        )
    return cast(RFQChains, audited)


class RFQGraphRuntime:
    """Own one SQLite checkpointer connection and a compiled RFQ graph."""

    def __init__(
        self,
        database_path: Path,
        *,
        chains: RFQChains,
        renderer: DocumentRenderer,
        model_name: str = "test-model",
    ) -> None:
        self.repository = SQLiteCaseRepository(database_path)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._connection)
        audited_chains = _with_audit(chains, self.repository, model_name)
        self.graph = build_rfq_graph(
            audited_chains, checkpointer=self._checkpointer, renderer=renderer
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def _config(case_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": case_id}}

    def start(self, case_id: str, initial_request: str) -> dict[str, Any]:
        self.repository.create_graph_case(case_id, initial_request)
        return self._invoke(case_id, {"case_id": case_id, "initial_request": initial_request})

    def continue_case(self, case_id: str) -> dict[str, Any]:
        self.repository.graph_case(case_id)
        return self._invoke(case_id, None)

    def resume(self, case_id: str, response: object) -> dict[str, Any]:
        self.repository.graph_case(case_id)
        return self._invoke(case_id, Command(resume=response))

    def _invoke(self, case_id: str, value: object) -> dict[str, Any]:
        token = _active_case_id.set(case_id)
        try:
            result = dict(self.graph.invoke(value, self._config(case_id)))
        except Exception as error:
            try:
                checkpoint = dict(self.graph.get_state(self._config(case_id)).values)
                self._sync_index(case_id, checkpoint)
            except Exception:
                pass
            if isinstance(error, OutputParserException):
                raise GraphExecutionError(
                    "The model returned an invalid structured response. "
                    f"Your case is saved; run `rfq-agent resume {case_id}` to retry."
                ) from error
            raise GraphExecutionError(
                f"Graph execution failed ({type(error).__name__})."
            ) from error
        finally:
            _active_case_id.reset(token)
        self._sync_index(case_id, result)
        return result

    def _sync_index(self, case_id: str, state: dict[str, Any]) -> None:
        status = str(state.get("status", "classifying"))
        document_path = state.get("document_path")
        self.repository.update_graph_case(
            case_id,
            status=status,
            document_path=str(document_path) if document_path else None,
        )
        if status == "completed" and document_path:
            self.repository.index_graph_document(
                case_id, int(state["draft_version"]), str(document_path)
            )

    @staticmethod
    def interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
        interrupts = result.get("__interrupt__", ())
        if not interrupts:
            return None
        return dict(interrupts[0].value)


def build_runtime(settings: Settings, *, output_dir: Path = Path("generated")) -> RFQGraphRuntime:
    """Build the production graph from LangChain chains and the configured database."""
    if not settings.llm_model.strip():
        raise ValueError("Set LLM_MODEL to an available model name.")
    if not settings.llm_api_key.get_secret_value():
        raise ValueError("Set LLM_API_KEY in your local environment or .env file.")
    model = build_chat_model(settings)
    chains = cast(RFQChains, build_rfq_chains(model))
    return RFQGraphRuntime(
        settings.rfq_database_path,
        chains=chains,
        renderer=GraphDocumentRenderer(output_dir),
        model_name=settings.llm_model,
    )

"""Build the LangGraph classification and interview workflow."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from rfq_agent.graph.nodes import (
    DocumentRenderer,
    DraftingNodes,
    InterviewChains,
    InterviewNodes,
    RFQChains,
)
from rfq_agent.graph.state import RFQGraphState


def build_interview_graph(
    chains: InterviewChains,
    *,
    checkpointer: Any,
) -> Any:
    """Compile the RFQ interview graph with caller-owned checkpoint storage."""
    nodes = InterviewNodes(chains)
    builder = StateGraph(RFQGraphState)
    builder.add_node("classify", nodes.classify)
    builder.add_node("confirm_type", nodes.confirm_type)
    builder.add_node("plan_question", nodes.plan_question)
    builder.add_node("ask_question", nodes.ask_question)
    builder.add_node("process_answer", nodes.process_answer)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "confirm_type")
    builder.add_conditional_edges(
        "confirm_type",
        nodes.route_after_type_confirmation,
        {"confirmed": "plan_question", "retry": "confirm_type"},
    )
    builder.add_conditional_edges(
        "plan_question",
        nodes.route_after_planning,
        {"ask": "ask_question", "ready": END, "plan": "plan_question"},
    )
    builder.add_edge("ask_question", "process_answer")
    builder.add_conditional_edges(
        "process_answer",
        nodes.route_after_answer,
        {"ask": "ask_question", "plan": "plan_question"},
    )
    return builder.compile(checkpointer=checkpointer)


def build_rfq_graph(
    chains: RFQChains,
    *,
    checkpointer: Any,
    renderer: DocumentRenderer,
) -> Any:
    """Compile the complete RFQ workflow with caller-owned persistence and rendering."""
    interview = InterviewNodes(chains)
    drafting = DraftingNodes(chains, renderer)
    builder = StateGraph(RFQGraphState)
    builder.add_node("classify", interview.classify)
    builder.add_node("confirm_type", interview.confirm_type)
    builder.add_node("plan_question", interview.plan_question)
    builder.add_node("ask_question", interview.ask_question)
    builder.add_node("process_answer", interview.process_answer)
    builder.add_node("draft", drafting.draft)
    builder.add_node("review", drafting.review)
    builder.add_node("request_revision", drafting.request_revision)
    builder.add_node("final_approval", drafting.final_approval)
    builder.add_node("generate_document", drafting.generate_document)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "confirm_type")
    builder.add_conditional_edges(
        "confirm_type",
        interview.route_after_type_confirmation,
        {"confirmed": "plan_question", "retry": "confirm_type"},
    )
    builder.add_conditional_edges(
        "plan_question",
        interview.route_after_planning,
        {"ask": "ask_question", "ready": "draft", "plan": "plan_question"},
    )
    builder.add_edge("ask_question", "process_answer")
    builder.add_conditional_edges(
        "process_answer",
        interview.route_after_answer,
        {"ask": "ask_question", "plan": "plan_question"},
    )
    builder.add_edge("draft", "review")
    builder.add_conditional_edges(
        "review",
        drafting.route_after_review,
        {"revision": "request_revision", "approval": "final_approval"},
    )
    builder.add_conditional_edges(
        "request_revision",
        drafting.route_after_revision,
        {
            "draft": "draft",
            "approval": "final_approval",
            "document": "generate_document",
            "revision": "request_revision",
        },
    )
    builder.add_conditional_edges(
        "final_approval",
        drafting.route_after_approval,
        {
            "document": "generate_document",
            "draft": "draft",
            "revision": "request_revision",
            "approval": "final_approval",
        },
    )
    builder.add_edge("generate_document", END)
    return builder.compile(checkpointer=checkpointer)

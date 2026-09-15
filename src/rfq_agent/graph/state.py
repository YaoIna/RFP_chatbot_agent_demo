"""State contract for the classification and interview portion of the RFQ graph."""

from typing import Literal, NotRequired, TypedDict

from rfq_agent.domain.models import SectionId


class ClassificationState(TypedDict):
    rfq_type: str
    confidence: float
    rationale: str
    alternatives: list[str]


class QuestionState(TypedDict):
    question_id: str
    wording: str
    target_field: str
    maps_to_sections: list[SectionId]


class InterviewAnswer(TypedDict):
    question_id: str
    question: str
    answer: str


class EffectiveFactState(TypedDict):
    fact_id: str
    value: str
    source_question_id: str


class SkippedTopic(TypedDict):
    question_id: str
    question: str
    target_field: str
    maps_to_sections: list[SectionId]
    partial_answer: NotRequired[str]
    missing_detail: NotRequired[str]


class DraftSectionState(TypedDict):
    section_id: SectionId
    heading: str
    body_markdown: str
    facts_used: list[str]
    missing_information_used: list[str]


class DraftState(TypedDict):
    sections: list[DraftSectionState]


class ReviewIssueState(TypedDict):
    issue_id: str
    section_id: SectionId | None
    description: str
    category: str


class ReviewState(TypedDict):
    issues: list[ReviewIssueState]


class RFQGraphState(TypedDict, total=False):
    case_id: str
    initial_request: str
    status: Literal[
        "classifying",
        "type_confirmation",
        "interviewing",
        "section_complete",
        "ready_to_draft",
        "drafting",
        "reviewing",
        "revision",
        "user_confirmation",
        "document_generation",
        "completed",
    ]
    classification: ClassificationState
    confirmed_type: str
    current_question: QuestionState | None
    pending_answer: str | None
    answer_fragments: list[str]
    clarification_message: str | None
    answers: dict[str, InterviewAnswer]
    answer_history: list[InterviewAnswer]
    effective_facts: dict[str, EffectiveFactState]
    fact_ledger_initialized: bool
    superseded_facts: list[EffectiveFactState]
    skipped_topics: list[SkippedTopic]
    asked_question_ids: list[str]
    section_cursor: int
    draft: DraftState
    draft_version: int
    review: ReviewState
    revision_instruction: str | None
    revision_fragments: list[str]
    pending_revision: bool
    approved: bool
    document_path: str

"""Structured outputs for the LangChain RFQ capabilities."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rfq_agent.domain.enums import RFQType
from rfq_agent.domain.models import SectionId


class ChainOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassificationDecision(ChainOutput):
    rfq_type: RFQType
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
    alternatives: list[RFQType] = Field(default_factory=list)


class OpenQuestion(ChainOutput):
    question_id: str = Field(min_length=1)
    wording: str = Field(min_length=1)
    target_field: str = Field(min_length=1)
    maps_to_sections: list[SectionId] = Field(min_length=1)


class PlanningDecision(ChainOutput):
    next_question: OpenQuestion | None = None
    sufficient: bool = False

    @model_validator(mode="after")
    def require_exactly_one_outcome(self) -> "PlanningDecision":
        if (self.next_question is not None) == self.sufficient:
            raise ValueError("Provide one next question or set sufficient=true.")
        return self


class FactUpdate(ChainOutput):
    fact_id: str = Field(min_length=1)
    value: str = Field(min_length=1)
    supersedes: list[str] = Field(default_factory=list)


class AnswerDecision(ChainOutput):
    outcome: Literal["accepted", "clarify_retain", "clarify_discard", "skipped"]
    clarification_message: str | None = None
    fact_updates: list[FactUpdate] = Field(default_factory=list)

    @model_validator(mode="after")
    def match_clarification_to_decision(self) -> "AnswerDecision":
        has_message = bool(self.clarification_message and self.clarification_message.strip())
        if self.outcome == "skipped":
            if has_message:
                raise ValueError("A skipped answer cannot also clarify an answer.")
            return self
        if self.outcome == "accepted" and has_message:
            raise ValueError("Accepted answers must not include a clarification message.")
        if self.outcome == "accepted" and not self.fact_updates:
            raise ValueError("Accepted answers must provide current facts.")
        if self.outcome.startswith("clarify_") and not has_message:
            raise ValueError("Clarification requires a clarification message.")
        return self


class FactSource(ChainOutput):
    source_id: str = Field(min_length=1)
    facts: list[FactUpdate] = Field(default_factory=list)


class FactLedgerDecision(ChainOutput):
    sources: list[FactSource] = Field(min_length=1)


class ResolvedSkippedTopic(ChainOutput):
    question_id: str = Field(min_length=1)
    facts: list[FactUpdate] = Field(min_length=1)


class TypeConfirmationDecision(ChainOutput):
    action: Literal["confirm", "change_type", "unclear"]
    selected_type: RFQType | None = None
    clarification_message: str | None = None

    @model_validator(mode="after")
    def match_type_to_action(self) -> "TypeConfirmationDecision":
        if self.action == "change_type" and self.selected_type is None:
            raise ValueError("A type change requires a selected RFQ type.")
        if self.action != "change_type" and self.selected_type is not None:
            raise ValueError("Only a type change may select an RFQ type.")
        if self.action == "unclear" and not self.clarification_message:
            raise ValueError("An unclear type reply requires a clarification message.")
        if self.action != "unclear" and self.clarification_message:
            raise ValueError("Only an unclear type reply may request clarification.")
        return self


class ReviewIntentDecision(ChainOutput):
    action: Literal["approve", "revise", "decline", "unclear", "cancel_pending_and_approve"]
    clarification_message: str | None = None

    @model_validator(mode="after")
    def match_message_to_action(self) -> "ReviewIntentDecision":
        if self.action == "unclear" and not self.clarification_message:
            raise ValueError("An unclear review reply requires a clarification message.")
        if self.action != "unclear" and self.clarification_message:
            raise ValueError("Only an unclear review reply may request clarification.")
        return self


class RevisionDecision(ChainOutput):
    outcome: Literal["apply", "clarify"]
    fact_updates: list[FactUpdate] = Field(default_factory=list)
    resolved_skipped_topics: list[ResolvedSkippedTopic] = Field(default_factory=list)
    draft_instruction: str | None = None
    clarification_message: str | None = None

    @model_validator(mode="after")
    def match_message_to_outcome(self) -> "RevisionDecision":
        if self.outcome == "clarify" and not self.clarification_message:
            raise ValueError("An unclear revision requires a clarification message.")
        if self.outcome == "apply" and self.clarification_message:
            raise ValueError("An applicable revision must not request clarification.")
        if self.outcome == "clarify" and self.draft_instruction:
            raise ValueError("An unclear revision cannot provide a draft instruction yet.")
        return self


class DraftSectionOutput(ChainOutput):
    section_id: SectionId
    heading: str = Field(min_length=1)
    body_markdown: str = Field(
        min_length=1,
        description=(
            "Section body using only plain paragraphs and dash-prefixed bullet lines; "
            "no headings, emphasis, or numbered-list Markdown."
        ),
    )
    facts_used: list[str] = Field(default_factory=list)
    missing_information_used: list[str] = Field(default_factory=list)


class DraftOutput(ChainOutput):
    sections: list[DraftSectionOutput] = Field(min_length=1)


class ReviewIssueOutput(ChainOutput):
    issue_id: str = Field(min_length=1)
    section_id: SectionId | None = None
    description: str = Field(min_length=1)
    category: Literal["accuracy", "completeness", "clarity", "consistency", "formatting"]
    requires_change: bool


class ReviewOutput(ChainOutput):
    draft_acceptable: bool
    issues: list[ReviewIssueOutput] = Field(default_factory=list)

    @property
    def actionable_issues(self) -> list[ReviewIssueOutput]:
        return [issue for issue in self.issues if issue.requires_change]

    @model_validator(mode="after")
    def match_acceptability_to_issues(self) -> "ReviewOutput":
        if self.draft_acceptable == bool(self.actionable_issues):
            raise ValueError("Draft acceptability must match the actionable review issues.")
        return self

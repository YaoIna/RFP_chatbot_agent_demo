"""LangGraph nodes for type confirmation and the open-question interview loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from langchain_core.runnables import Runnable
from langgraph.types import interrupt

from rfq_agent.chains.schemas import (
    AnswerDecision,
    ClassificationDecision,
    DraftOutput,
    FactLedgerDecision,
    FactUpdate,
    OpenQuestion,
    PlanningDecision,
    ReviewIntentDecision,
    ReviewOutput,
    RevisionDecision,
    TypeConfirmationDecision,
)
from rfq_agent.domain.enums import RFQType
from rfq_agent.domain.models import SectionId
from rfq_agent.graph.state import (
    ClassificationState,
    DraftSectionState,
    DraftState,
    EffectiveFactState,
    InterviewAnswer,
    QuestionState,
    ReviewIssueState,
    ReviewState,
    RFQGraphState,
    SkippedTopic,
)

SECTION_IDS: tuple[SectionId, ...] = (
    "background",
    "scope",
    "service_levels",
    "vendor_response",
    "pricing",
    "evaluation",
    "timeline",
)


class FactCollisionError(ValueError):
    """A model tried to overwrite an unrelated current fact without declaring it."""

    def __init__(self, fact_id: str) -> None:
        self.fact_id = fact_id
        super().__init__(f"Fact ID collision for {fact_id!r}; replacement requires supersedes.")


def _collision_feedback(error: FactCollisionError) -> str:
    return (
        f"The fact_id {error.fact_id!r} already belongs to a different current value. "
        "If the new value corrects that fact, include the ID in supersedes. If it is an "
        "unrelated fact, return a distinct qualified fact_id."
    )


def _merge_fact_updates(
    state: RFQGraphState, updates: list[FactUpdate], source_question_id: str
) -> dict[str, Any]:
    active = dict(state.get("effective_facts", {}))
    superseded = list(state.get("superseded_facts", []))
    for update in updates:
        existing = active.get(update.fact_id)
        if existing is not None and update.fact_id not in update.supersedes:
            if existing["value"] == update.value:
                continue
            raise FactCollisionError(update.fact_id)
        for prior_id in set(update.supersedes):
            prior = active.pop(prior_id, None)
            if prior is not None and prior["value"] != update.value:
                superseded.append(prior)
        active[update.fact_id] = EffectiveFactState(
            fact_id=update.fact_id,
            value=update.value,
            source_question_id=source_question_id,
        )
    return {"effective_facts": active, "superseded_facts": superseded}


class InterviewChains(Protocol):
    classifier: Runnable[dict[str, Any], ClassificationDecision]
    fact_ledger_initializer: Runnable[dict[str, Any], FactLedgerDecision]
    planner: Runnable[dict[str, Any], PlanningDecision]
    answer_validator: Runnable[dict[str, Any], AnswerDecision]
    type_interpreter: Runnable[dict[str, Any], TypeConfirmationDecision]


class RFQChains(InterviewChains, Protocol):
    review_interpreter: Runnable[dict[str, Any], ReviewIntentDecision]
    revision_interpreter: Runnable[dict[str, Any], RevisionDecision]
    writer: Runnable[dict[str, Any], DraftOutput]
    reviewer: Runnable[dict[str, Any], ReviewOutput]


class DocumentRenderer(Protocol):
    def render(self, state: RFQGraphState) -> Path: ...


def _ensure_fact_ledger(
    chains: InterviewChains, state: RFQGraphState
) -> tuple[RFQGraphState, dict[str, Any]]:
    if state.get("fact_ledger_initialized"):
        return state, {}
    sources = [
        {"source_id": "initial_request", "text": state["initial_request"]},
        *[
            {"source_id": answer["question_id"], "text": answer["answer"]}
            for answer in state.get("answers", {}).values()
        ],
    ]
    payload = {
        "rfq_type": RFQType(state["confirmed_type"]).value,
        "sources": sources,
        "effective_facts": list(state.get("effective_facts", {}).values()),
    }
    expected = [source["source_id"] for source in sources]
    feedback: str | None = None
    for attempt in range(2):
        decision = FactLedgerDecision.model_validate(
            chains.fact_ledger_initializer.invoke(
                payload if feedback is None else {**payload, "output_feedback": feedback}
            )
        )
        actual = [source.source_id for source in decision.sources]
        if actual != expected:
            raise ValueError("Fact ledger initializer must return every source once, in order.")
        working = state.copy()
        try:
            for source in decision.sources:
                merged = _merge_fact_updates(working, source.facts, source.source_id)
                working["effective_facts"] = merged["effective_facts"]
                working["superseded_facts"] = merged["superseded_facts"]
            break
        except FactCollisionError as error:
            if attempt:
                raise
            feedback = _collision_feedback(error)
    updates: dict[str, Any] = {
        "effective_facts": working.get("effective_facts", {}),
        "superseded_facts": working.get("superseded_facts", []),
        "fact_ledger_initialized": True,
    }
    merged_state = state.copy()
    merged_state.update(updates)  # type: ignore[typeddict-item]
    return merged_state, updates


class InterviewNodes:
    def __init__(self, chains: InterviewChains) -> None:
        self._chains = chains

    def classify(self, state: RFQGraphState) -> dict[str, Any]:
        decision = ClassificationDecision.model_validate(
            self._chains.classifier.invoke({"description": state["initial_request"]})
        )
        classification: ClassificationState = {
            "rfq_type": decision.rfq_type.value,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "alternatives": [alternative.value for alternative in decision.alternatives],
        }
        return {
            "classification": classification,
            "status": "type_confirmation",
            "answers": state.get("answers", {}),
            "answer_history": state.get("answer_history", []),
            "effective_facts": state.get("effective_facts", {}),
            "superseded_facts": state.get("superseded_facts", []),
            "skipped_topics": state.get("skipped_topics", []),
            "asked_question_ids": state.get("asked_question_ids", []),
            "section_cursor": state.get("section_cursor", 0),
            "current_question": None,
            "pending_answer": None,
            "answer_fragments": [],
            "clarification_message": None,
            "revision_fragments": state.get("revision_fragments", []),
            "pending_revision": state.get("pending_revision", False),
            "fact_ledger_initialized": state.get("fact_ledger_initialized", False),
        }

    def confirm_type(self, state: RFQGraphState) -> dict[str, Any]:
        classification = ClassificationDecision.model_validate(state["classification"])
        proposed_type = classification.rfq_type
        prompt = {
            "kind": "type_confirmation",
            "proposed_type": proposed_type.value,
            "confidence": classification.confidence,
            "rationale": classification.rationale,
            "alternatives": [alternative.value for alternative in classification.alternatives],
            "clarification_message": state.get("clarification_message"),
        }
        response = interrupt(prompt)
        confirmed_type = self._parse_confirmed_type(response, proposed_type)
        if confirmed_type is None:
            return {
                "status": "type_confirmation",
                "clarification_message": "Confirm the type or name a supported RFQ type.",
            }
        return {
            "confirmed_type": confirmed_type.value,
            "status": "interviewing",
            "clarification_message": None,
        }

    def plan_question(self, state: RFQGraphState) -> dict[str, Any]:
        state, ledger_updates = _ensure_fact_ledger(self._chains, state)
        answers = list(state.get("answers", {}).values())
        effective_facts = list(state.get("effective_facts", {}).values())
        answered_topics = [
            {"question_id": answer["question_id"], "question": answer["question"]}
            for answer in answers
        ]
        skipped_topics = state.get("skipped_topics", [])
        cursor = state.get("section_cursor", 0)
        target_section = SECTION_IDS[cursor] if cursor < len(SECTION_IDS) else None
        decision = PlanningDecision.model_validate(
            self._chains.planner.invoke(
                {
                    "rfq_type": RFQType(state["confirmed_type"]).value,
                    "facts": effective_facts,
                    "answered_topics": answered_topics,
                    "skipped_topics": skipped_topics,
                    "section_ids": list(SECTION_IDS),
                    "target_section": target_section,
                }
            )
        )
        if decision.sufficient:
            if target_section is not None:
                return {
                    **ledger_updates,
                    "status": "section_complete",
                    "section_cursor": cursor + 1,
                    "current_question": None,
                    "pending_answer": None,
                    "answer_fragments": [],
                    "clarification_message": None,
                }
            return {
                **ledger_updates,
                "status": "ready_to_draft",
                "current_question": None,
                "pending_answer": None,
                "answer_fragments": [],
                "clarification_message": None,
            }

        question = decision.next_question
        if question is None:  # guarded by PlanningDecision
            raise ValueError("Planner must return one question or sufficient=true.")
        if target_section is not None and target_section not in question.maps_to_sections:
            raise ValueError(f"Planner question for {target_section} must map to that section.")
        asked_ids = state.get("asked_question_ids", [])
        question_id = question.question_id
        suffix = 2
        while question_id in asked_ids:
            question_id = f"{question.question_id}_{suffix}"
            suffix += 1
        question_state: QuestionState = {
            "question_id": question_id,
            "wording": question.wording,
            "target_field": question.target_field,
            "maps_to_sections": list(question.maps_to_sections),
        }
        return {
            **ledger_updates,
            "status": "interviewing",
            "current_question": question_state,
            "asked_question_ids": [*asked_ids, question_id],
            "pending_answer": None,
            "answer_fragments": [],
            "clarification_message": None,
        }

    def ask_question(self, state: RFQGraphState) -> dict[str, Any]:
        question = self._current_question(state)
        response = interrupt(
            {
                "kind": "question",
                "question_id": question.question_id,
                "wording": question.wording,
                "clarification_message": state.get("clarification_message"),
            }
        )
        while not isinstance(response, str):
            response = interrupt(
                {
                    "kind": "question",
                    "question_id": question.question_id,
                    "wording": question.wording,
                    "clarification_message": "Enter a text answer or type skip.",
                }
            )
        return {"pending_answer": response}

    def process_answer(self, state: RFQGraphState) -> dict[str, Any]:
        state, ledger_updates = _ensure_fact_ledger(self._chains, state)
        question = self._current_question(state)
        raw_answer = state.get("pending_answer")
        if raw_answer is None:
            raise ValueError("No answer is pending for the current question.")

        known_facts: Any = {
            "effective_facts": list(state.get("effective_facts", {}).values()),
            "superseded_facts": state.get("superseded_facts", []),
            "answer_history": state.get("answer_history", []),
        }

        payload = {
            "rfq_type": RFQType(state["confirmed_type"]).value,
            "known_facts": known_facts,
            "question": question.wording,
            "answer_fragments": state.get("answer_fragments", []),
            "clarification_message": state.get("clarification_message"),
            "answer": raw_answer,
        }
        feedback: str | None = None
        for attempt in range(2):
            decision = AnswerDecision.model_validate(
                self._chains.answer_validator.invoke(
                    payload if feedback is None else {**payload, "output_feedback": feedback}
                )
            )
            try:
                fact_updates = _merge_fact_updates(
                    state,
                    decision.fact_updates,
                    question.question_id,
                )
                break
            except FactCollisionError as error:
                if attempt:
                    raise
                feedback = _collision_feedback(error)
        history: InterviewAnswer = {
            "question_id": question.question_id,
            "question": question.wording,
            "answer": raw_answer,
        }
        turn_updates: dict[str, Any] = {
            **ledger_updates,
            **fact_updates,
            "answer_history": [*state.get("answer_history", []), history],
        }
        if decision.outcome == "skipped":
            skipped: SkippedTopic = {
                "question_id": question.question_id,
                "question": question.wording,
                "target_field": question.target_field,
                "maps_to_sections": list(question.maps_to_sections),
            }
            fragments = [
                *state.get("answer_fragments", []),
                *([raw_answer] if decision.fact_updates else []),
            ]
            if fragments:
                skipped["partial_answer"] = "\n".join(fragments)
                if clarification := state.get("clarification_message"):
                    skipped["missing_detail"] = clarification
            return {
                **turn_updates,
                "skipped_topics": [*state.get("skipped_topics", []), skipped],
                "current_question": None,
                "pending_answer": None,
                "answer_fragments": [],
                "clarification_message": None,
            }
        replaces_known_fact = any(update.supersedes for update in decision.fact_updates)
        previous_fragments = [] if replaces_known_fact else state.get("answer_fragments", [])
        if decision.outcome.startswith("clarify_"):
            return {
                **turn_updates,
                "pending_answer": None,
                "answer_fragments": [
                    *previous_fragments,
                    *([raw_answer] if decision.outcome == "clarify_retain" else []),
                ],
                "clarification_message": decision.clarification_message,
            }

        answer: InterviewAnswer = {
            "question_id": question.question_id,
            "question": question.wording,
            "answer": "\n".join([*previous_fragments, raw_answer]),
        }
        answers = {**state.get("answers", {}), question.question_id: answer}
        return {
            **turn_updates,
            "answers": answers,
            "current_question": None,
            "pending_answer": None,
            "answer_fragments": [],
            "clarification_message": None,
        }

    @staticmethod
    def route_after_type_confirmation(state: RFQGraphState) -> str:
        return "confirmed" if state["status"] == "interviewing" else "retry"

    @staticmethod
    def route_after_planning(state: RFQGraphState) -> str:
        if state["status"] == "section_complete":
            return "plan"
        return "ready" if state["status"] == "ready_to_draft" else "ask"

    @staticmethod
    def route_after_answer(state: RFQGraphState) -> str:
        return "ask" if state.get("current_question") is not None else "plan"

    @staticmethod
    def _current_question(state: RFQGraphState) -> OpenQuestion:
        question = state.get("current_question")
        if question is None:
            raise ValueError("The graph has no current interview question.")
        return OpenQuestion.model_validate(question)

    def _parse_confirmed_type(self, response: object, proposed_type: RFQType) -> RFQType | None:
        if response is True:
            return proposed_type
        if isinstance(response, RFQType):
            return response
        if isinstance(response, str):
            decision = TypeConfirmationDecision.model_validate(
                self._chains.type_interpreter.invoke(
                    {
                        "proposed_type": proposed_type.value,
                        "answer": response,
                    }
                )
            )
            if decision.action == "confirm":
                return proposed_type
            if decision.action == "change_type":
                return decision.selected_type
        return None


class DraftingNodes:
    """Draft and approve an RFQ after the interview is complete."""

    def __init__(self, chains: RFQChains, renderer: DocumentRenderer) -> None:
        self._chains = chains
        self._renderer = renderer

    def draft(self, state: RFQGraphState) -> dict[str, Any]:
        state, ledger_updates = _ensure_fact_ledger(self._chains, state)
        facts: dict[str, Any] = {"effective_facts": list(state.get("effective_facts", {}).values())}
        revision_context: dict[str, Any] | None = None
        if state.get("draft"):
            revision_context = {
                "instruction": state.get("revision_instruction"),
                "previous_draft": state.get("draft"),
                "review": state.get("review"),
            }
        skipped = state.get("skipped_topics", [])
        result = DraftOutput.model_validate(
            self._chains.writer.invoke(
                {
                    "rfq_type": RFQType(state["confirmed_type"]).value,
                    "facts": facts,
                    "revision_context": revision_context,
                    "missing_information": skipped,
                    "section_ids": list(SECTION_IDS),
                }
            )
        )
        draft = self._validated_draft(result, state)
        return {
            **ledger_updates,
            "status": "reviewing",
            "draft": draft,
            "draft_version": state.get("draft_version", 0) + 1,
            "review": {"issues": []},
            "revision_instruction": None,
            "revision_fragments": [],
            "pending_revision": False,
            "approved": False,
        }

    def review(self, state: RFQGraphState) -> dict[str, Any]:
        state, ledger_updates = _ensure_fact_ledger(self._chains, state)
        facts: dict[str, Any] = {"effective_facts": list(state.get("effective_facts", {}).values())}
        result = ReviewOutput.model_validate(
            self._chains.reviewer.invoke(
                {
                    "rfq_type": RFQType(state["confirmed_type"]).value,
                    "draft": state["draft"],
                    "facts": facts,
                    "missing_information": state.get("skipped_topics", []),
                }
            )
        )
        actionable = result.actionable_issues
        issue_ids = [issue.issue_id for issue in actionable]
        if len(issue_ids) != len(set(issue_ids)):
            raise ValueError("Reviewer issue IDs must be unique within a review.")
        issues: list[ReviewIssueState] = []
        for issue in actionable:
            if issue.section_id is not None and issue.section_id not in SECTION_IDS:
                raise ValueError("Reviewer issue references an unknown section.")
            issues.append(
                {
                    "issue_id": issue.issue_id,
                    "section_id": issue.section_id,
                    "description": issue.description,
                    "category": issue.category,
                }
            )
        review: ReviewState = {"issues": issues}
        return {
            **ledger_updates,
            "review": review,
            "status": "revision" if issues else "user_confirmation",
        }

    def request_revision(self, state: RFQGraphState) -> dict[str, Any]:
        prompt = {
            "kind": "revision",
            "draft_version": state["draft_version"],
            "issues": state["review"]["issues"],
            "clarification_message": state.get("clarification_message"),
        }
        response = interrupt(prompt)
        decision = self._interpret_review_intent("revision", response, state)
        if decision.action == "unclear":
            return {
                "status": "revision",
                "revision_fragments": self._revision_fragments(state, response),
                "clarification_message": (
                    decision.clarification_message
                    or "Approve the current draft or describe the revision required."
                ),
            }
        if decision.action in {"approve", "cancel_pending_and_approve"}:
            if state.get("pending_revision") and decision.action != "cancel_pending_and_approve":
                return {
                    "status": "revision",
                    "clarification_message": state.get("clarification_message")
                    or "Clarify the pending change or explicitly cancel it before approval.",
                }
            return {
                "status": "document_generation",
                "approved": True,
                "clarification_message": None,
                "revision_fragments": [],
                "pending_revision": False,
            }
        if decision.action == "revise":
            return self._extract_revision("revision", state, response)
        return {
            "status": "revision",
            "approved": False,
            "clarification_message": None,
        }

    def final_approval(self, state: RFQGraphState) -> dict[str, Any]:
        prompt = {
            "kind": "final_approval",
            "draft_version": state["draft_version"],
            "draft": state["draft"],
            "review": state["review"],
            "clarification_message": state.get("clarification_message"),
        }
        response = interrupt(prompt)
        decision = self._interpret_review_intent("final_approval", response, state)
        if decision.action == "unclear":
            return {
                "status": "user_confirmation",
                "revision_fragments": self._revision_fragments(state, response),
                "clarification_message": (
                    decision.clarification_message
                    or "Approve the current draft or describe the revision required."
                ),
            }
        if decision.action in {"approve", "cancel_pending_and_approve"}:
            if state.get("pending_revision") and decision.action != "cancel_pending_and_approve":
                return {
                    "status": "user_confirmation",
                    "clarification_message": state.get("clarification_message")
                    or "Clarify the pending change or explicitly cancel it before approval.",
                }
            return {
                "status": "document_generation",
                "approved": True,
                "clarification_message": None,
                "revision_fragments": [],
                "pending_revision": False,
            }
        if decision.action == "decline":
            return {
                "status": "revision",
                "approved": False,
                "clarification_message": None,
                "revision_fragments": [],
                "pending_revision": False,
            }
        return self._extract_revision("final_approval", state, response)

    def _extract_revision(
        self, stage: str, state: RFQGraphState, response: object
    ) -> dict[str, Any]:
        state, ledger_updates = _ensure_fact_ledger(self._chains, state)
        if not isinstance(response, str):
            return {
                **ledger_updates,
                "status": "revision" if stage == "revision" else "user_confirmation",
                "pending_revision": True,
                "clarification_message": "Describe the revision in text.",
            }
        payload = {
            "effective_facts": list(state.get("effective_facts", {}).values()),
            "superseded_facts": state.get("superseded_facts", []),
            "skipped_topics": state.get("skipped_topics", []),
            "review_issues": state.get("review", {}).get("issues", []),
            "revision_fragments": state.get("revision_fragments", []),
            "clarification_message": state.get("clarification_message"),
            "answer": response,
        }
        feedback: str | None = None
        for attempt in range(2):
            decision = RevisionDecision.model_validate(
                self._chains.revision_interpreter.invoke(
                    payload if feedback is None else {**payload, "output_feedback": feedback}
                )
            )
            if decision.outcome == "clarify":
                return {
                    **ledger_updates,
                    "status": "revision" if stage == "revision" else "user_confirmation",
                    "pending_revision": True,
                    "revision_fragments": self._revision_fragments(state, response),
                    "clarification_message": decision.clarification_message,
                }
            try:
                return self._revision_updates(state, decision)
            except FactCollisionError as error:
                if attempt:
                    raise
                feedback = _collision_feedback(error)
        raise AssertionError("Revision retry loop must return or raise.")

    @staticmethod
    def _revision_fragments(state: RFQGraphState, response: object) -> list[str]:
        fragments = list(state.get("revision_fragments", []))
        return [*fragments, response] if isinstance(response, str) else fragments

    @staticmethod
    def _revision_updates(state: RFQGraphState, decision: RevisionDecision) -> dict[str, Any]:
        skipped = state.get("skipped_topics", [])
        resolved_ids = [topic.question_id for topic in decision.resolved_skipped_topics]
        resolved = set(resolved_ids)
        if len(resolved) != len(resolved_ids) or resolved - {
            topic["question_id"] for topic in skipped
        }:
            raise ValueError("Review correction references an unknown skipped topic.")
        revision_facts = [
            *decision.fact_updates,
            *(fact for topic in decision.resolved_skipped_topics for fact in topic.facts),
        ]
        return {
            **_merge_fact_updates(state, revision_facts, f"revision_v{state['draft_version']}"),
            "skipped_topics": [topic for topic in skipped if topic["question_id"] not in resolved],
            "status": "drafting",
            "approved": False,
            "revision_instruction": decision.draft_instruction,
            "revision_fragments": [],
            "pending_revision": False,
            "fact_ledger_initialized": True,
            "clarification_message": None,
        }

    def _interpret_review_intent(
        self, stage: str, response: object, state: RFQGraphState
    ) -> ReviewIntentDecision:
        if response is True:
            return ReviewIntentDecision(action="approve")
        if response is False:
            return ReviewIntentDecision(action="decline")
        if not isinstance(response, str) or not response.strip():
            return ReviewIntentDecision(
                action="unclear", clarification_message="Approve or describe a revision."
            )
        return ReviewIntentDecision.model_validate(
            self._chains.review_interpreter.invoke(
                {
                    "stage": stage,
                    "prompt": "Approve the draft or describe the revision required.",
                    "pending_revision": state.get("pending_revision", False),
                    "revision_fragments": state.get("revision_fragments", []),
                    "clarification_message": state.get("clarification_message"),
                    "answer": response,
                }
            )
        )

    def generate_document(self, state: RFQGraphState) -> dict[str, Any]:
        if not state.get("approved"):
            raise ValueError("Document generation requires explicit user approval.")
        path = self._renderer.render(state).resolve()
        return {"status": "completed", "document_path": str(path)}

    @staticmethod
    def route_after_review(state: RFQGraphState) -> str:
        return "revision" if state["status"] == "revision" else "approval"

    @staticmethod
    def route_after_revision(state: RFQGraphState) -> str:
        if state["status"] == "document_generation":
            return "document"
        if state["status"] == "revision":
            return "revision"
        return "draft" if state["status"] == "drafting" else "approval"

    @staticmethod
    def route_after_approval(state: RFQGraphState) -> str:
        if state["status"] == "document_generation":
            return "document"
        if state["status"] == "drafting":
            return "draft"
        if state["status"] == "revision":
            return "revision"
        return "approval"

    @staticmethod
    def _validated_draft(result: DraftOutput, state: RFQGraphState) -> DraftState:
        if [section.section_id for section in result.sections] != list(SECTION_IDS):
            raise ValueError("Draft sections must match the required section IDs and order.")
        known_facts = set(state.get("effective_facts", {}))
        skipped = {topic["question_id"]: topic for topic in state.get("skipped_topics", [])}
        sections: list[DraftSectionState] = []
        used_missing: list[str] = []
        for output in result.sections:
            if len(output.facts_used) != len(set(output.facts_used)):
                raise ValueError("Draft section repeats a fact ID.")
            if len(output.missing_information_used) != len(set(output.missing_information_used)):
                raise ValueError("Draft section repeats a missing information ID.")
            if set(output.facts_used) - known_facts:
                raise ValueError("Draft section references an unknown fact ID.")
            if set(output.missing_information_used) - set(skipped):
                raise ValueError("Draft section references an unknown missing information ID.")
            for topic_id in output.missing_information_used:
                if output.section_id not in skipped[topic_id]["maps_to_sections"]:
                    raise ValueError("Draft places missing information in an unmapped section.")
            used_missing.extend(output.missing_information_used)
            sections.append(
                {
                    "section_id": output.section_id,
                    "heading": output.heading,
                    "body_markdown": output.body_markdown,
                    "facts_used": list(output.facts_used),
                    "missing_information_used": list(output.missing_information_used),
                }
            )
        if sorted(used_missing) != sorted(skipped):
            raise ValueError("Draft must cite every missing information ID exactly once.")
        return {"sections": sections}

"""Build LangChain prompt pipelines for RFQ graph nodes."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from rfq_agent.chains.schemas import (
    AnswerDecision,
    ClassificationDecision,
    DraftOutput,
    FactLedgerDecision,
    PlanningDecision,
    ReviewIntentDecision,
    ReviewOutput,
    RevisionDecision,
    TypeConfirmationDecision,
)

PromptInput = dict[str, Any]


@dataclass(frozen=True)
class RFQChains:
    classifier: Runnable[PromptInput, ClassificationDecision]
    fact_ledger_initializer: Runnable[PromptInput, FactLedgerDecision]
    planner: Runnable[PromptInput, PlanningDecision]
    answer_validator: Runnable[PromptInput, AnswerDecision]
    type_interpreter: Runnable[PromptInput, TypeConfirmationDecision]
    review_interpreter: Runnable[PromptInput, ReviewIntentDecision]
    revision_interpreter: Runnable[PromptInput, RevisionDecision]
    writer: Runnable[PromptInput, DraftOutput]
    reviewer: Runnable[PromptInput, ReviewOutput]


def _system_prompt(name: str) -> str:
    return (Path(__file__).parent / "prompts" / f"{name}.md").read_text().strip()


def _chain(
    model: BaseChatModel,
    *,
    prompt_name: str,
    human_template: str,
    schema: type[ClassificationDecision]
    | type[FactLedgerDecision]
    | type[PlanningDecision]
    | type[AnswerDecision]
    | type[TypeConfirmationDecision]
    | type[ReviewIntentDecision]
    | type[RevisionDecision]
    | type[DraftOutput]
    | type[ReviewOutput],
) -> Runnable[PromptInput, Any]:
    system_message = SystemMessage(
        content=(
            f"{_system_prompt(prompt_name)}\n\n"
            "Return JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), separators=(',', ':'))}"
        )
    )
    prompt = ChatPromptTemplate.from_messages([system_message, ("human", human_template)])
    structured = model.with_structured_output(schema, method="json_mode")

    def invoke(payload: PromptInput, config: RunnableConfig) -> Any:
        formatted = prompt.invoke(
            {"output_feedback": None, "revision_context": None, **payload},
            config=config,
        )
        try:
            return structured.invoke(formatted, config=config)
        except OutputParserException as error:
            messages = formatted.to_messages()
            if error.llm_output:
                messages.append(AIMessage(content=error.llm_output))
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous response could not be parsed as the required JSON. "
                        "Re-evaluate the original request and return one valid JSON object "
                        "matching the schema. Keep the user's facts unchanged."
                    )
                )
            )
            return structured.invoke(ChatPromptValue(messages=messages), config=config)

    return RunnableLambda(invoke)


def build_rfq_chains(model: BaseChatModel) -> RFQChains:
    """Create the structured chains consumed by the RFQ graph."""
    return RFQChains(
        classifier=_chain(
            model,
            prompt_name="classification",
            human_template="RFQ request:\n{description}",
            schema=ClassificationDecision,
        ),
        fact_ledger_initializer=_chain(
            model,
            prompt_name="fact-ledger",
            human_template=(
                "RFQ type: {rfq_type}\nSources to extract: {sources}\n"
                "Existing current facts: {effective_facts}\n"
                "Structured-output feedback from a previous attempt: {output_feedback}"
            ),
            schema=FactLedgerDecision,
        ),
        planner=_chain(
            model,
            prompt_name="planner",
            human_template=(
                "RFQ type: {rfq_type}\nCurrent facts: {facts}\n"
                "Answered topics: {answered_topics}\n"
                "Skipped topics: {skipped_topics}\nRFQ sections: {section_ids}\n"
                "Section currently being assessed: {target_section}"
            ),
            schema=PlanningDecision,
        ),
        answer_validator=_chain(
            model,
            prompt_name="answer-validation",
            human_template=(
                "RFQ type: {rfq_type}\nCurrent facts: {known_facts}\nQuestion: {question}\n"
                "Earlier replies to this question: {answer_fragments}\n"
                "Clarification requested: {clarification_message}\n"
                "Latest reply: {answer}\n"
                "Structured-output feedback from a previous attempt: {output_feedback}"
            ),
            schema=AnswerDecision,
        ),
        type_interpreter=_chain(
            model,
            prompt_name="type-confirmation",
            human_template=("Proposed RFQ type: {proposed_type}\nUser reply: {answer}"),
            schema=TypeConfirmationDecision,
        ),
        review_interpreter=_chain(
            model,
            prompt_name="review-intent",
            human_template=(
                "Stage: {stage}\nCurrent prompt: {prompt}\n"
                "A revision is awaiting clarification: {pending_revision}\n"
                "Clarification requested: {clarification_message}\n"
                "Earlier replies to this revision: {revision_fragments}\n"
                "User reply: {answer}"
            ),
            schema=ReviewIntentDecision,
        ),
        revision_interpreter=_chain(
            model,
            prompt_name="revision-extraction",
            human_template=(
                "Current facts: {effective_facts}\nSuperseded facts: {superseded_facts}\n"
                "Previously skipped topics: {skipped_topics}\nReview issues: {review_issues}\n"
                "Earlier replies to this revision: {revision_fragments}\n"
                "Clarification requested: {clarification_message}\n"
                "Latest revision reply: {answer}\n"
                "Structured-output feedback from a previous attempt: {output_feedback}"
            ),
            schema=RevisionDecision,
        ),
        writer=_chain(
            model,
            prompt_name="writer",
            human_template=(
                "RFQ type: {rfq_type}\nKnown facts: {facts}\n"
                "Revision context: {revision_context}\n"
                "Missing information: {missing_information}\nRFQ sections: {section_ids}"
            ),
            schema=DraftOutput,
        ),
        reviewer=_chain(
            model,
            prompt_name="reviewer",
            human_template=(
                "RFQ type: {rfq_type}\nDraft: {draft}\nKnown facts: {facts}\n"
                "Missing information: {missing_information}"
            ),
            schema=ReviewOutput,
        ),
    )

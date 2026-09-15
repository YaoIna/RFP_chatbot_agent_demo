"""Construct the LangChain chat model from application settings."""

from langchain_openai import ChatOpenAI

from rfq_agent.settings import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Create an OpenAI-compatible LangChain model without making a request."""
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        temperature=0,
    )

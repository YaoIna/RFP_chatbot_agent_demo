"""Small shared type aliases used at LangChain/LangGraph boundaries."""

from typing import Literal

SectionId = Literal[
    "background",
    "scope",
    "service_levels",
    "vendor_response",
    "pricing",
    "evaluation",
    "timeline",
]

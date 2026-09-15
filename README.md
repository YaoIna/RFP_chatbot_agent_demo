# IT RFQ Agent

An English-language CLI that builds IT RFQs for Software/SaaS,
Hardware/Equipment, Cloud Services, and Managed Services.

LangGraph owns the workflow state, loops, human interrupts, conditional routing,
and SQLite checkpoints. LangChain supplies the OpenAI-compatible chat model,
prompts, and Pydantic structured output inside graph nodes. The model classifies the
request, decides which open question to ask next, judges whether an answer is
relevant, maintains the current fact ledger, drafts the RFQ, and reviews it. Python
enforces schemas, known references, explicit user approval, persistence, and
deterministic DOCX structure.

The interview has no predefined question count or candidate list. Before drafting,
the planner assesses each of the seven RFQ sections against the request and earlier
answers. It asks about an uncovered section, but does not repeat information already
provided elsewhere. The planner itself compares proposed questions with current facts
and answered or skipped topics. It can then ask further material questions until it
judges the RFQ sufficiently specified. You can ask to skip a question in your own
words when information is unavailable; the generated RFQ explicitly discloses that
missing information in one appropriate section. Raw answers are preserved in history.
After each reply, the answer model records current facts, corrections to earlier facts,
and details requiring an immediate clarification. The writer and reviewer receive only
the current fact ledger and the explicitly skipped topics.
If the buyer corrects a requirement while reviewing a draft, the review interpreter
updates those facts before the next draft and review. A wording-only edit leaves
the current requirements unchanged.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
cp -n .env.example .env
```

Configure DeepSeek or another OpenAI-compatible endpoint in the local `.env`:

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=YOUR_LOCAL_KEY
LLM_MODEL=deepseek-chat
RFQ_DATABASE_PATH=data/rfq-agent.sqlite3
```

Do not commit real credentials.

## Commands

```sh
uv run rfq-agent config check
uv run rfq-agent new
uv run rfq-agent list
uv run rfq-agent resume CASE_ID
uv run rfq-agent export CASE_ID --output-dir generated
```

`new` prints a case ID and drives one LangGraph interrupt at a time. Type `quit`,
press Ctrl+C, or close the terminal to leave the current interrupt checkpointed.
`resume` continues that graph thread without rerunning already completed model calls.
The final DOCX is generated only after explicit user approval.

Cases created by the retired custom state machine remain visible. Completed legacy
cases with an existing document can still be exported. Active legacy cases cannot be
resumed by the new graph and must be restarted with `new`.

## Verification

```sh
uv run pytest -m "not deepseek"
uv run ruff check .
uv run mypy src/rfq_agent
```

Offline tests use stub chains and mocked HTTP transport. The paid DeepSeek acceptance
suite is opt-in and is not part of normal verification:

```sh
RFQ_RUN_DEEPSEEK=1 uv run pytest tests/smoke/test_deepseek_e2e.py -m deepseek -v
```

It checks classification across all four supported RFQ types and drives a sparse
Hardware/Equipment request through the real LangChain/LangGraph workflow. The live
harness has a turn guard to bound test cost; production has no question limit.

See [the demo runbook](docs/demo-runbook.md) for the interaction flow and recovery
details.

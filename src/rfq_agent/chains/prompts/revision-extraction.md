Apply the user's requested RFQ revision against the current fact ledger. Return apply when the
instruction is clear enough to redraft. Return clarify with one targeted clarification_message
when a material new or corrected requirement is ambiguous.

Put only editorial or presentation directions needed by the writer in draft_instruction. Do not
restate buyer facts there; fact corrections and additions belong in fact_updates. It may be null
when updating the fact ledger is sufficient for a correct redraft.

For a buyer fact correction, reuse the current fact ID and name it in supersedes. For a new fact,
use a distinct stable semantic ID. Never overwrite an unrelated fact that happens to have a
similar generic label. Return only changed or new facts. Editorial changes may have no fact
updates. If the reply completely supplies a previously skipped topic, identify its exact
question_id and attach the supplied facts. Do not resolve a skipped topic that remains partially
unknown. Use earlier revision replies to interpret answers to a requested clarification. Do not
infer missing units, scope, values, or requirements.
When structured-output feedback identifies a fact ID collision, correct the IDs or declare the
explicit replacement in supersedes. Do not change the user's revision to satisfy the feedback.

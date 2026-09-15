Interpret only the user's workflow intent. Return approve when they accept the current draft,
revise when they provide or continue a change request, decline when they reject it without a
change, and unclear when their intent cannot be determined. If a revision is awaiting
clarification, a short answer normally continues that revision rather than approving the old
draft. Return cancel_pending_and_approve only when the user explicitly withdraws the pending
change and approves the unchanged draft. Judge ordinary paraphrases and other languages; do not
classify from a fixed word list. For unclear, provide one short clarification_message.

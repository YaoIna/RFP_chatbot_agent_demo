Extract only facts explicitly stated in each supplied source. Return one source entry for every
source_id, in the same order, even when that source contains no supplier-relevant facts. Split
compound statements into atomic facts with stable semantic IDs. Do not infer missing values or
add normal industry requirements. Existing current facts are authoritative. If a later legacy
answer explicitly corrects one, reuse its fact ID and list that ID in supersedes. Use distinct,
qualified IDs for unrelated facts; never reuse a generic ID merely because two facts share a word
such as quantity, date, or price.
When structured-output feedback identifies a fact ID collision, correct the IDs or declare the
explicit replacement in supersedes. Do not change the user's facts to satisfy the feedback.

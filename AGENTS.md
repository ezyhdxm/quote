# Coding and research agreements

For all code added or edited in this repository:

- Identify domain/algorithm blocks with `# CORE LOGIC: STEP 1`, `STEP 2`, etc., numbered in reading order within the function/cell. Split by meaningful calculation stages.
- Each CORE block must contain at most 10 physical nonblank, noncomment code lines. Count every line of a multiline expression. Do not squeeze new statements onto a line or mislabel business logic as plotting/cache code to evade the limit.
- Before each CORE block, write a complete concrete `# Input:` / `# Output:` example, like a LeetCode example. Use actual numbers, timestamps, arrays or small records; state required inputs and the resulting intermediate/output values. Match the example to this block, not merely the whole function. Explain rounding if applicable.
- Explain non-obvious techniques with `# Trick:`: broadcasting/index alignment, as-of boundaries, prefix sums, nanosecond units, mutation/aliasing, missing-value masks, deterministic ordering, etc.
- Clearly label non-core regions with their responsibility: `PLOTTING LOGIC`, `CACHEING LOGIC` (this spelling), `FILE IO LOGIC`, `UI LOGIC`, `SETUP LOGIC`, `CONFIGURATION LOGIC`, `TEST LOGIC`, or a precise equivalent.
- Keep notebook source cells and matching Python source synchronized. Preserve existing outputs/kernel state during comment-only updates.
- For comment-only work, verify AST equivalence. Check the CORE length/example convention with `python tools/check_logic_comments.py`. Review example arithmetic and algorithm classification manually; the checker does not prove them.
- Preserve checked-cache identity across pure annotation changes; real executable changes must invalidate it. Do not weaken source/content validation merely to reuse cache.

BondCliQ constraints: three-month traded-bond universe; known time; existing TRACE/multiplier and BASE14/target/anchor; no automatic removal of zero/negative spreads, unknown quantity, multiple prices or crossing. Preserve completed Step5 work. Research documents/data are local only; push code only when requested. Tests and synthetic examples are not predictive-gain evidence.

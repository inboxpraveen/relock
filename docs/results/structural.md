# RelapseBench - structural mode

Model-free and deterministic: every system receives the identical scripted feedback stream, and we measure which rule variants end up in the memory context an agent would see. This verifies the mechanism; it is **not** a measurement of model behaviour.

Cells are counts over families x seeds. Seeds only re-sample the distractor items, so the counts are not independent samples and no confidence intervals are given.

| load | k | memory budget (tokens) | distractor items | padding tokens | stale + injected text in history |
|---|---|---|---|---|---|
| L0 | 5 | 800 | 0 | 0 | no |
| L1 | 4 | 400 | 20 | 1500 | yes |
| L2 | 3 | 200 | 60 | 4000 | yes |
| L3 | 2 | 110 | 120 | 8000 | yes |

### Superseded rule in context

| system | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| append-notes | 90% (27/30) | 90% (27/30) | 50% (15/30) | 40% (12/30) |
| append-clause | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| ace-style | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| append-gated | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| overwrite-ungated | 0% (0/30) | 0% (0/30) | 0% (0/30) | 0% (0/30) |
| relock | 0% (0/30) | 0% (0/30) | 0% (0/30) | 0% (0/30) |

### Poisoned rule in context

| system | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| append-notes | 100% (30/30) | 100% (30/30) | 100% (30/30) | 57% (17/30) |
| append-clause | 100% (30/30) | 100% (30/30) | 100% (30/30) | 60% (18/30) |
| ace-style | 100% (30/30) | 100% (30/30) | 100% (30/30) | 80% (24/30) |
| append-gated | 0% (0/30) | 0% (0/30) | 0% (0/30) | 0% (0/30) |
| overwrite-ungated | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| relock | 0% (0/30) | 0% (0/30) | 0% (0/30) | 0% (0/30) |

### Conflicting rules in context

| system | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| append-notes | 100% (30/30) | 100% (30/30) | 100% (30/30) | 77% (23/30) |
| append-clause | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| ace-style | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| append-gated | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| overwrite-ungated | 10% (3/30) | 10% (3/30) | 10% (3/30) | 10% (3/30) |
| relock | 0% (0/30) | 0% (0/30) | 0% (0/30) | 0% (0/30) |

### Correction in context

| system | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| append-notes | 100% (30/30) | 100% (30/30) | 90% (27/30) | 77% (23/30) |
| append-clause | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| ace-style | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| append-gated | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| overwrite-ungated | 10% (3/30) | 10% (3/30) | 10% (3/30) | 10% (3/30) |
| relock | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |

### Only the correction in context

| system | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| append-notes | 0% (0/30) | 0% (0/30) | 0% (0/30) | 0% (0/30) |
| append-clause | 0% (0/30) | 0% (0/30) | 0% (0/30) | 0% (0/30) |
| ace-style | 0% (0/30) | 0% (0/30) | 0% (0/30) | 0% (0/30) |
| append-gated | 0% (0/30) | 0% (0/30) | 0% (0/30) | 0% (0/30) |
| overwrite-ungated | 0% (0/30) | 0% (0/30) | 0% (0/30) | 0% (0/30) |
| relock | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |

### Learning cost (trusted errors before the correction took effect)

| system | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| append-notes | 1.00 (100% learned) | 1.00 (100% learned) | 1.00 (100% learned) | 1.00 (100% learned) |
| append-clause | 1.00 (100% learned) | 1.00 (100% learned) | 1.00 (100% learned) | 1.00 (100% learned) |
| ace-style | 1.00 (100% learned) | 1.00 (100% learned) | 1.00 (100% learned) | 1.00 (100% learned) |
| append-gated | 2.00 (100% learned) | 2.00 (100% learned) | 2.00 (100% learned) | 2.00 (100% learned) |
| overwrite-ungated | 1.00 (100% learned) | 1.00 (100% learned) | 1.00 (100% learned) | 1.00 (100% learned) |
| relock | 2.00 (100% learned) | 2.00 (100% learned) | 2.00 (100% learned) | 2.00 (100% learned) |

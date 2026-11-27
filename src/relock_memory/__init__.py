"""relock - agent memory that unlocks on recall, rewrites on surprise, and re-locks.

Most agent memories fix mistakes by *appending* ("don't do X"), which leaves the
old instruction retrievable; under long context or compaction the old default
wins again. relock borrows the shape of memory reconsolidation instead:

1. **Reactivation = retrieval.** Only items recalled in the current episode are editable.
2. **Explicit prediction.** Before acting on an item, the harness records what it should produce.
3. **Prediction error = write trigger.** Only a *trusted* mismatch (typed tool error,
   verifier failure, authenticated user) counts; transient errors and untrusted text never do.
4. **Overwrite in place.** Enough evidence rewrites the clause that fired as a small
   delta; the old version is superseded (kept for audit and rollback), never left active.
5. **Re-lock.** When the episode closes, the editable set closes with it.
"""

from __future__ import annotations

from . import checks
from .checks import Check
from .delta import DeltaError, Op
from .episode import Episode
from .integrations.toolkit import Toolkit
from .llm import OpenAICompatible
from .memory import Memory
from .policy import WritePolicy
from .retrieval import BM25Retriever, EmbeddingRetriever, HybridRetriever, Retriever
from .rewriters import (
    CallableRewriter,
    HintRewriter,
    LLMRewriter,
    Proposal,
    Rewriter,
    RewriteRequest,
)
from .signals import ErrorClass, Signal, Source, ToolError, TransientToolError, classify_exception
from .stores import SQLiteStore, SQLStore, open_store
from .types import (
    Clause,
    ConflictError,
    Decision,
    EpisodeClosedError,
    Expectation,
    Hit,
    Item,
    NotFoundError,
    NotLabileError,
    Observation,
    Recall,
    RelockError,
    Verdict,
)

__version__ = "0.2.1"

__all__ = [
    "BM25Retriever",
    "CallableRewriter",
    "Check",
    "Clause",
    "ConflictError",
    "Decision",
    "DeltaError",
    "EmbeddingRetriever",
    "Episode",
    "EpisodeClosedError",
    "ErrorClass",
    "Expectation",
    "HintRewriter",
    "Hit",
    "HybridRetriever",
    "Item",
    "LLMRewriter",
    "Memory",
    "NotFoundError",
    "NotLabileError",
    "Observation",
    "Op",
    "OpenAICompatible",
    "Proposal",
    "Recall",
    "RelockError",
    "Retriever",
    "RewriteRequest",
    "Rewriter",
    "SQLStore",
    "SQLiteStore",
    "Signal",
    "Source",
    "ToolError",
    "Toolkit",
    "TransientToolError",
    "Verdict",
    "WritePolicy",
    "__version__",
    "checks",
    "classify_exception",
    "open_store",
]

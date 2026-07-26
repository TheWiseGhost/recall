"""Cost estimation for a run.

The rule this module exists to enforce: **never report a price Recall does not
know.** A paid model with no configured price reports ``None``, not ``0.0``.
Reporting a fabricated zero would make an expensive configuration look free,
which is exactly the sort of wrong conclusion the project is built to prevent.

Only query-side embedding is estimated here. Ingestion cost is a property of the
corpus and the chunking strategy rather than of a retrieval run, and every run
in a sweep shares the same index — charging each of them for the same ingest
would multiply one cost by the size of the sweep.
"""

from __future__ import annotations

from recall.core.embeddings.base import EmbeddingModelInfo
from recall.core.evaluation.models import CostEstimate

# Providers that run on the caller's own hardware. For these, an absent price
# genuinely means free. Extending this set is how a new local provider avoids
# being reported as "not priced".
LOCAL_PROVIDERS = frozenset({"hash", "sentence_transformers"})


def estimate_embedding_cost(
    model: EmbeddingModelInfo | None,
    tokens: int,
) -> CostEstimate:
    """Estimate the embedding spend for ``tokens`` tokens.

    Three distinguishable outcomes, never collapsed into one number:

    * a priced model -> the computed cost
    * a local provider with no price -> ``0.0``, because local is free
    * anything else with no price -> ``None`` and a note, because Recall does
      not know and will not guess
    """
    if model is None:
        return CostEstimate(usd=None, embedded_tokens=tokens, note="no embedding model in use")

    price = model.cost_per_million_tokens
    if price is not None:
        return CostEstimate(
            usd=round(tokens * price / 1_000_000, 8),
            embedded_tokens=tokens,
            model=model.key,
            cost_per_million_tokens=price,
            note="estimated from the model's configured price",
        )

    if model.provider in LOCAL_PROVIDERS:
        return CostEstimate(
            usd=0.0,
            embedded_tokens=tokens,
            model=model.key,
            note=f"{model.provider} runs locally; no per-token charge",
        )

    return CostEstimate(
        usd=None,
        embedded_tokens=tokens,
        model=model.key,
        note=(
            f"no price known for {model.key}; cost is not estimated rather than assumed to be zero"
        ),
    )

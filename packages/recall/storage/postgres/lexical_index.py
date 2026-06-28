"""BM25 over PostgreSQL full-text search.

This is Okapi BM25 as specified, not ``ts_rank_cd`` wearing its name:

.. code-block:: text

    score(D, Q) = sum over t in Q of
        IDF(t) * ( f(t,D) * (k1 + 1) )
               / ( f(t,D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

``ts_rank_cd`` measures cover density — how tightly the query terms cluster in
the text. It has no IDF term and no length saturation, so it cannot express
"this match is on a rare word" or "this chunk is long, discount its term
counts". Shipping it under the BM25 name would silently corrupt every
retrieval comparison Recall exists to make.

Every quantity comes out of PostgreSQL's own inverted index:

===========  =============================================================
``f(t,D)``   ``array_length(positions, 1)`` from ``unnest(content_tsv)``
``|D|``      ``chunks.content_length`` (migration 0002)
``n(t)``     count of candidate chunks containing ``t``
``N``        number of chunks in the filtered collection
``avgdl``    mean ``content_length`` over the filtered collection
===========  =============================================================

Statistics are scoped to the *filtered* collection, so a filtered search is
scored against the corpus it actually searched rather than the whole database.

Two properties worth knowing:

* ``n(t)`` computed over the candidate set is exact, not an approximation:
  candidates are every chunk matching at least one query term, so any chunk
  containing ``t`` is already among them.
* PostgreSQL stores at most 256 positions per lexeme in a tsvector, so
  ``f(t,D)`` saturates at 256. For chunk-sized text this is unreachable; for a
  whole-book chunk it would flatten term frequency. Documented rather than
  worked around.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import ColumnElement, Float, cast, func, literal, select, true
from sqlalchemy.dialects.postgresql import REGCONFIG, TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from recall.core.models import SearchFilters, SearchResult, SourceType
from recall.storage.postgres.engine import session_scope
from recall.storage.postgres.filters import build_document_predicates
from recall.storage.postgres.models import TEXT_SEARCH_CONFIG, ChunkRow, DocumentRow

# Defaults from Robertson & Zaragoza, and what Lucene and Elasticsearch ship.
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


def _regconfig(name: str) -> ColumnElement[Any]:
    """``'english'::regconfig`` — the two-argument text search functions need it."""
    return cast(literal(name), REGCONFIG)


def _lexemes_of(text_expression: ColumnElement[Any]) -> Any:
    """``unnest(tsvector)`` as a table of ``(lexeme, positions, weights)``."""
    return func.unnest(text_expression).table_valued("lexeme", "positions", "weights")


def _tsquery_literal(lexeme: ColumnElement[Any]) -> ColumnElement[Any]:
    """Quote one lexeme as a tsquery token that is never re-parsed.

    This deliberately does *not* use ``to_tsquery``. ``to_tsquery`` runs the
    text-search parser over its input even for quoted tokens, so a lexeme the
    parser would split — ``o'brien``, ``a.com/p?q=1&r=2``, any URL or path —
    comes back as a phrase of fragments that appear nowhere in the index, and
    the match is silently lost.

    Casting text straight to ``tsquery`` uses the type's input function, which
    treats a quoted token as one opaque lexeme. Inside such a token the only
    escapes are ``''`` for a quote and ``\\\\`` for a backslash, so both are
    doubled here; the round trip is then exact.
    """
    escaped = func.replace(func.replace(lexeme, "\\", "\\\\"), "'", "''")
    return literal("'") + escaped + literal("'")


class PostgresBM25Index:
    """Implements :class:`recall.core.ports.LexicalIndex` on PostgreSQL FTS.

    ``k1`` controls term-frequency saturation (how quickly extra occurrences
    stop helping) and ``b`` how strongly scores are normalised by chunk length;
    ``b=0`` disables length normalisation entirely.
    """

    name = "bm25"

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self._sessions = session_factory
        self.k1 = k1
        self.b = b

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0 or not query.strip():
            return []

        statement = self._build_statement(query, top_k=top_k, filters=filters)
        async with session_scope(self._sessions) as session:
            rows = (await session.execute(statement)).all()

        return [
            SearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                content=row.content,
                score=round(float(row.score), 6),
                rank=rank,
                metadata=dict(row.chunk_metadata or {}),
                document_title=row.document_title,
                document_uri=row.document_uri,
                source_type=SourceType(row.source_type),
                retriever=self.name,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    # -- query construction ------------------------------------------------

    def _build_statement(self, query: str, *, top_k: int, filters: SearchFilters | None) -> Any:
        """Assemble the whole scorer as one statement.

        One round trip on purpose: splitting it would mean shipping candidate
        IDs back and forth, and computing corpus statistics against a corpus
        that had moved in between.
        """
        predicates = build_document_predicates(filters)

        # 1. The query's lexemes, produced by the *same* analyzer that built
        #    content_tsv, so stemming and stopwords can never drift apart.
        analyzed = func.to_tsvector(_regconfig(TEXT_SEARCH_CONFIG), literal(query))
        query_lexemes = _lexemes_of(analyzed)
        query_terms = select(query_lexemes.c.lexeme.label("lexeme")).cte("query_terms")

        # 2. Those lexemes OR-ed into a tsquery, which is what the GIN index on
        #    content_tsv can answer. OR, not AND: BM25 ranks by accumulated
        #    evidence, and requiring every term would be a recall cutoff that
        #    dense retrieval does not have, making the two incomparable.
        tsquery = select(
            cast(
                func.string_agg(_tsquery_literal(query_terms.c.lexeme), literal(" | ")),
                TSQUERY,
            ).label("q")
        ).cte("query_tsquery")

        # 3. Candidates: chunks matching at least one term, filtered in SQL.
        candidates = (
            select(
                ChunkRow.id.label("chunk_id"),
                ChunkRow.document_id.label("document_id"),
                ChunkRow.content.label("content"),
                ChunkRow.meta.label("chunk_metadata"),
                ChunkRow.content_tsv.label("content_tsv"),
                cast(ChunkRow.content_length, Float).label("content_length"),
                DocumentRow.title.label("document_title"),
                DocumentRow.uri.label("document_uri"),
                DocumentRow.source_type.label("source_type"),
            )
            .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
            .where(ChunkRow.content_tsv.op("@@")(select(tsquery.c.q).scalar_subquery()))
        )
        for predicate in predicates:
            candidates = candidates.where(predicate)
        matches = candidates.cte("matches")

        # 4. Collection statistics over everything the filters admit — not
        #    just the candidates. avgdl must describe the corpus searched.
        corpus = (
            select(
                cast(func.count(), Float).label("n_docs"),
                cast(func.coalesce(func.avg(ChunkRow.content_length), 0.0), Float).label("avgdl"),
            )
            .select_from(ChunkRow)
            .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
        )
        for predicate in predicates:
            corpus = corpus.where(predicate)
        corpus_cte = corpus.cte("corpus")

        n_docs = select(corpus_cte.c.n_docs).scalar_subquery()
        avgdl = select(corpus_cte.c.avgdl).scalar_subquery()

        # 5. f(t, D) for every (candidate, query term) pair. Spelled LATERAL
        #    explicitly: a set-returning function in FROM is implicitly lateral
        #    to PostgreSQL, but only the keyword tells SQLAlchemy the two are
        #    correlated rather than a cartesian product.
        chunk_lexemes = _lexemes_of(matches.c.content_tsv).lateral()
        term_freqs = (
            select(
                matches.c.chunk_id.label("chunk_id"),
                matches.c.content_length.label("content_length"),
                chunk_lexemes.c.lexeme.label("lexeme"),
                cast(
                    func.coalesce(func.array_length(chunk_lexemes.c.positions, 1), 1), Float
                ).label("tf"),
            )
            .select_from(matches.join(chunk_lexemes, true()))
            .where(chunk_lexemes.c.lexeme.in_(select(query_terms.c.lexeme)))
            .cte("term_freqs")
        )

        # 6. n(t). Exact: every chunk containing t is a candidate.
        doc_freqs = (
            select(
                term_freqs.c.lexeme.label("lexeme"),
                cast(func.count(func.distinct(term_freqs.c.chunk_id)), Float).label("df"),
            )
            .group_by(term_freqs.c.lexeme)
            .cte("doc_freqs")
        )

        # 7. The formula itself.
        idf = func.ln(1.0 + (n_docs - doc_freqs.c.df + 0.5) / (doc_freqs.c.df + 0.5))
        # coalesce/nullif guards an empty collection; it contributes no rows
        # anyway, but division by zero would abort the statement.
        length_norm = (
            1.0
            - self.b
            + self.b * (term_freqs.c.content_length / func.coalesce(func.nullif(avgdl, 0.0), 1.0))
        )
        saturation = (term_freqs.c.tf * (self.k1 + 1.0)) / (term_freqs.c.tf + self.k1 * length_norm)

        scored = (
            select(
                term_freqs.c.chunk_id.label("chunk_id"),
                func.sum(idf * saturation).label("score"),
            )
            .select_from(term_freqs.join(doc_freqs, doc_freqs.c.lexeme == term_freqs.c.lexeme))
            .group_by(term_freqs.c.chunk_id)
            .cte("scored")
        )

        return (
            select(
                scored.c.chunk_id,
                matches.c.document_id,
                matches.c.content,
                matches.c.chunk_metadata,
                matches.c.document_title,
                matches.c.document_uri,
                matches.c.source_type,
                scored.c.score,
            )
            .select_from(scored.join(matches, matches.c.chunk_id == scored.c.chunk_id))
            # chunk_id breaks ties deterministically, so repeated runs of the
            # same experiment produce the same ranking.
            .order_by(scored.c.score.desc(), scored.c.chunk_id)
            .limit(top_k)
        )

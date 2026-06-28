"""BM25 against a live PostgreSQL, cross-checked against a reference.

The load-bearing test here is :meth:`TestAgainstReference.test_scores_match`. It
recomputes BM25 in Python from the same inverted index PostgreSQL exposes —
lexemes and their positions, read straight out of ``content_tsv`` — and asserts
the SQL agrees to within floating-point noise.

The reference shares exactly one assumption with the implementation: how
PostgreSQL tokenizes. Everything BM25-specific — IDF, saturation, length
normalisation, the collection statistics — is computed independently, so an
error in the SQL algebra cannot hide behind a matching error in the check.
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict

import pytest
import pytest_asyncio
from sqlalchemy import text

from recall.core.models import Chunk, Document, SearchFilters, SourceType
from recall.core.retrieval.lexical import BM25Retriever
from recall.storage.postgres.lexical_index import PostgresBM25Index
from recall.storage.postgres.storage import Storage

pytestmark = pytest.mark.integration

K1 = 1.2
B = 0.75

# Term frequencies and lengths here are deliberately lopsided: "authentication"
# is rare, "service" is everywhere, and the documents differ in length several
# times over. A scorer without IDF or without length normalisation ranks these
# differently, so the assertions below can actually tell them apart.
CORPUS: list[tuple[str, SourceType, str, str]] = [
    (
        "auth.md",
        SourceType.FILESYSTEM,
        "Authentication",
        "Authentication uses a bearer token. The service verifies the token signature.",
    ),
    (
        "tokens.md",
        SourceType.FILESYSTEM,
        "Tokens",
        (
            "Token rotation happens hourly. The service issues a token, the service caches "
            "the token, and the service revokes the token when the session ends. Every "
            "service in the platform depends on this service for token validation, and the "
            "service emits a metric whenever a token is issued or revoked by the service."
        ),
    ),
    (
        "deploy.md",
        SourceType.FILESYSTEM,
        "Deployment",
        "Rolling deployment replaces replicas one at a time. The service stays available.",
    ),
    (
        "guide.pdf",
        SourceType.PDF,
        "Operator guide",
        "The service exposes a health endpoint. Authentication is not required for it.",
    ),
]


async def write_chunk(storage: Storage, document: Document, chunk: Chunk) -> uuid.UUID:
    """Persist one document and one chunk, with no vectors.

    Deliberately not ``index_document``: BM25 reads nothing from
    ``chunk_embeddings``, and an index with no vectors in it is the sharpest
    proof of that.
    """
    await storage.documents.upsert(document)
    await storage.chunks.replace_for_document(document.id, [chunk])
    return chunk.id


async def _index_corpus(storage: Storage) -> dict[str, uuid.UUID]:
    """Write one chunk per document, so chunk-level scoring is legible."""
    ids: dict[str, uuid.UUID] = {}
    for source_id, source_type, title, content in CORPUS:
        document = Document.create(
            source_id=source_id,
            source_type=source_type,
            title=title,
            content=content,
            uri=f"file:///{source_id}",
        )
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=document.id,
            content=content,
            position=0,
            token_count=len(content.split()),
            checksum=document.checksum,
        )
        ids[source_id] = await write_chunk(storage, document, chunk)
    return ids


# --- the reference implementation -------------------------------------------


async def reference_scores(
    storage: Storage,
    query: str,
    *,
    k1: float = K1,
    b: float = B,
    source_types: list[SourceType] | None = None,
) -> dict[uuid.UUID, float]:
    """Okapi BM25, computed in Python from PostgreSQL's inverted index."""
    where = ""
    params: dict[str, object] = {"q": query}
    if source_types:
        where = "WHERE d.source_type = ANY(:types)"
        params["types"] = [t.value for t in source_types]

    async with storage.sessions() as session:
        query_terms = {
            row[0]
            for row in (
                await session.execute(
                    text("SELECT lexeme FROM unnest(to_tsvector('english', :q))"), params
                )
            ).all()
        }
        postings = (
            await session.execute(
                text(
                    f"""
                    SELECT c.id, l.lexeme, coalesce(array_length(l.positions, 1), 1)
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    CROSS JOIN unnest(c.content_tsv) AS l
                    {where}
                    """  # `where` is a fixed literal chosen above, never user input
                ),
                params,
            )
        ).all()

    term_freq: dict[uuid.UUID, dict[str, float]] = defaultdict(dict)
    length: dict[uuid.UUID, float] = defaultdict(float)
    for chunk_id, lexeme, freq in postings:
        term_freq[chunk_id][lexeme] = float(freq)
        length[chunk_id] += float(freq)

    n_docs = float(len(length))
    if not n_docs:
        return {}
    avgdl = sum(length.values()) / n_docs

    doc_freq: dict[str, float] = defaultdict(float)
    for freqs in term_freq.values():
        for lexeme in freqs:
            if lexeme in query_terms:
                doc_freq[lexeme] += 1.0

    scores: dict[uuid.UUID, float] = {}
    for chunk_id, freqs in term_freq.items():
        total = 0.0
        for lexeme in query_terms & freqs.keys():
            tf = freqs[lexeme]
            df = doc_freq[lexeme]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            norm = 1.0 - b + b * (length[chunk_id] / avgdl)
            total += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)
        if total:
            scores[chunk_id] = total
    return scores


# --- fixtures ---------------------------------------------------------------


@pytest_asyncio.fixture
async def indexed(storage: Storage) -> dict[str, uuid.UUID]:
    return await _index_corpus(storage)


@pytest.fixture
def bm25(storage: Storage) -> PostgresBM25Index:
    return PostgresBM25Index(storage.sessions, k1=K1, b=B)


# --- tests ------------------------------------------------------------------


class TestGeneratedColumns:
    async def test_content_length_counts_indexed_terms(self, storage: Storage) -> None:
        """|D| is post-stemming, post-stopword token count — not word count."""
        await _index_corpus(storage)
        async with storage.sessions() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT content_length, "
                        "recall_tsvector_length(content_tsv), "
                        "array_length(string_to_array(content, ' '), 1) "
                        "FROM chunks"
                    )
                )
            ).all()
        assert rows
        for content_length, from_tsv, words in rows:
            assert content_length == from_tsv
            # Stopwords ("the", "a", "is", "for") are dropped, so it is strictly
            # smaller than the raw word count.
            assert 0 < content_length < words

    async def test_is_maintained_without_the_orm(self, storage: Storage) -> None:
        """A generated column must stay correct for writers that bypass Python."""
        await _index_corpus(storage)
        async with storage.sessions() as session:
            await session.execute(
                text("UPDATE chunks SET content = 'authentication authentication authentication'")
            )
            await session.commit()
            lengths = (
                await session.execute(text("SELECT DISTINCT content_length FROM chunks"))
            ).all()
        assert [row[0] for row in lengths] == [3]


class TestAgainstReference:
    @pytest.mark.parametrize(
        "query",
        [
            "authentication",
            "service token",
            "how does authentication verify a token",
            "rolling deployment replicas",
        ],
    )
    async def test_scores_match(
        self,
        storage: Storage,
        bm25: PostgresBM25Index,
        indexed: dict[str, uuid.UUID],
        query: str,
    ) -> None:
        expected = await reference_scores(storage, query)
        results = await bm25.search(query, top_k=100)

        assert {r.chunk_id for r in results} == set(expected)
        for result in results:
            assert result.score == pytest.approx(expected[result.chunk_id], abs=1e-5)

    async def test_ranking_matches(
        self, storage: Storage, bm25: PostgresBM25Index, indexed: dict[str, uuid.UUID]
    ) -> None:
        query = "service token authentication"
        expected = await reference_scores(storage, query)
        ranked = sorted(expected, key=lambda cid: (-expected[cid], str(cid)))
        results = await bm25.search(query, top_k=100)
        assert [r.chunk_id for r in results] == ranked

    async def test_statistics_are_scoped_to_the_filtered_corpus(
        self, storage: Storage, bm25: PostgresBM25Index, indexed: dict[str, uuid.UUID]
    ) -> None:
        """N, avgdl and n(t) must describe the corpus actually searched."""
        filters = SearchFilters(source_types=[SourceType.FILESYSTEM])
        expected = await reference_scores(
            storage, "authentication service", source_types=[SourceType.FILESYSTEM]
        )
        results = await bm25.search("authentication service", top_k=100, filters=filters)

        assert results
        assert all(r.source_type is SourceType.FILESYSTEM for r in results)
        for result in results:
            assert result.score == pytest.approx(expected[result.chunk_id], abs=1e-5)

        # And the filtered scores genuinely differ from the unfiltered ones,
        # which is what proves the statistics were recomputed rather than
        # inherited from the whole database.
        unfiltered = {
            r.chunk_id: r.score for r in await bm25.search("authentication service", top_k=100)
        }
        assert any(
            unfiltered[r.chunk_id] != pytest.approx(r.score, abs=1e-9)
            for r in results
            if r.chunk_id in unfiltered
        )


class TestBM25Properties:
    async def test_idf_prefers_the_rare_term(
        self, bm25: PostgresBM25Index, indexed: dict[str, uuid.UUID]
    ) -> None:
        """ "authentication" is in 2 of 4 chunks; "service" is in all 4."""
        rare = await bm25.search("authentication", top_k=10)
        common = await bm25.search("service", top_k=10)
        assert max(r.score for r in rare) > max(r.score for r in common)

    async def test_length_normalisation_discounts_long_chunks(
        self, storage: Storage, indexed: dict[str, uuid.UUID]
    ) -> None:
        """tokens.md repeats "token" far more, but is far longer.

        With b=0 the raw term counts win. With b=0.75 length normalisation
        pulls it back. If the two agree, |D| is not reaching the formula.
        """
        no_norm = await PostgresBM25Index(storage.sessions, k1=K1, b=0.0).search("token", top_k=10)
        with_norm = await PostgresBM25Index(storage.sessions, k1=K1, b=1.0).search(
            "token", top_k=10
        )

        long_chunk = indexed["tokens.md"]
        rank_without = [r.chunk_id for r in no_norm].index(long_chunk)
        rank_with = [r.chunk_id for r in with_norm].index(long_chunk)
        assert rank_without < rank_with

    async def test_saturation_is_sublinear_in_term_frequency(
        self, storage: Storage, bm25: PostgresBM25Index, indexed: dict[str, uuid.UUID]
    ) -> None:
        """Ten occurrences must not score ten times one occurrence."""
        results = {r.chunk_id: r.score for r in await bm25.search("token", top_k=10)}
        async with storage.sessions() as session:
            freqs = dict(
                (
                    await session.execute(
                        text(
                            "SELECT c.id, coalesce(array_length(l.positions, 1), 1) "
                            "FROM chunks c CROSS JOIN unnest(c.content_tsv) l "
                            "WHERE l.lexeme = 'token'"
                        )
                    )
                ).all()
            )
        ordered = sorted(freqs, key=lambda cid: freqs[cid])
        low, high = ordered[0], ordered[-1]
        assert freqs[high] > freqs[low]
        assert results[high] / results[low] < freqs[high] / freqs[low]


class TestQueryHandling:
    async def test_stopword_only_query_returns_nothing(
        self, bm25: PostgresBM25Index, indexed: dict[str, uuid.UUID]
    ) -> None:
        assert await bm25.search("the and of a", top_k=10) == []

    async def test_blank_query_short_circuits(self, bm25: PostgresBM25Index) -> None:
        assert await bm25.search("   ", top_k=10) == []

    async def test_non_positive_top_k_short_circuits(self, bm25: PostgresBM25Index) -> None:
        assert await bm25.search("token", top_k=0) == []

    async def test_top_k_limits_results(
        self, bm25: PostgresBM25Index, indexed: dict[str, uuid.UUID]
    ) -> None:
        assert len(await bm25.search("service token", top_k=2)) == 2

    async def test_stems_the_query_like_the_index(
        self, bm25: PostgresBM25Index, indexed: dict[str, uuid.UUID]
    ) -> None:
        """ "deployments" must find "deployment": one analyzer, both sides."""
        results = await bm25.search("deployments", top_k=10)
        assert [r.chunk_id for r in results] == [indexed["deploy.md"]]

    async def test_lexemes_with_punctuation_survive(self, storage: Storage) -> None:
        """The reason this does not use ``to_tsquery``.

        ``to_tsquery`` re-parses even quoted tokens, shattering a URL lexeme
        into fragments that appear nowhere in the index — a silent miss.
        """
        document = Document.create(
            source_id="urls.md",
            source_type=SourceType.FILESYSTEM,
            title="URLs",
            content="Fetch http://example.com/p?q=1&r=2 for the manifest.",
            uri="file:///urls.md",
        )
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=document.id,
            content=document.content,
            position=0,
            token_count=8,
            checksum=document.checksum,
        )
        await write_chunk(storage, document, chunk)

        index = PostgresBM25Index(storage.sessions, k1=K1, b=B)
        results = await index.search("http://example.com/p?q=1&r=2", top_k=10)
        assert [r.chunk_id for r in results] == [chunk.id]


class TestRetrieverIntegration:
    async def test_search_through_the_retriever(
        self, storage: Storage, indexed: dict[str, uuid.UUID]
    ) -> None:
        retriever = BM25Retriever(index=PostgresBM25Index(storage.sessions))
        results = await retriever.search("authentication token", top_k=3)
        assert results
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        assert all(r.retriever == "bm25" for r in results)
        assert all(r.document_title and r.document_uri for r in results)

    async def test_carries_chunk_metadata(
        self, storage: Storage, indexed: dict[str, uuid.UUID]
    ) -> None:
        retriever = BM25Retriever(index=PostgresBM25Index(storage.sessions))
        result = (await retriever.search("authentication", top_k=1))[0]
        assert isinstance(result.metadata, dict)
        assert result.source_type is not None

"""BM25 retriever contract, and the SQL builder's shape.

The arithmetic itself is verified against a reference implementation in
``tests/integration/test_lexical_index.py``, where the real query runs against
PostgreSQL. What is testable without a database is the retriever's contract and
the parts of the SQL that are easy to get quietly wrong.
"""

from __future__ import annotations

import uuid

from sqlalchemy.dialects import postgresql

from recall.core.models import Chunk, Document, SearchFilters, SourceType
from recall.core.retrieval import create_retriever, retriever_registry
from recall.core.retrieval.lexical import BM25Retriever
from recall.pipeline.search import SearchService
from recall.storage.postgres.lexical_index import PostgresBM25Index

from tests.conftest import FakeLexicalIndex


def populate(index: FakeLexicalIndex) -> None:
    corpus = [
        ("auth.md", SourceType.FILESYSTEM, "Authentication", "bearer tokens verify the signature"),
        ("deploy.txt", SourceType.FILESYSTEM, "Deployment", "rolling deployments replace replicas"),
        ("guide.pdf", SourceType.PDF, "Guide", "the tokens scope is matched against the endpoint"),
    ]
    for source_id, source_type, title, content in corpus:
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
            checksum="x",
        )
        index.add(chunk, document)


class TestRegistration:
    def test_bm25_is_registered(self) -> None:
        assert "bm25" in retriever_registry

    def test_resolves_through_the_registry(self) -> None:
        retriever = create_retriever("bm25", index=FakeLexicalIndex())
        assert isinstance(retriever, BM25Retriever)
        assert retriever.name == "bm25"


class TestBM25Retriever:
    async def test_returns_ranked_results(self) -> None:
        index = FakeLexicalIndex()
        populate(index)
        results = await BM25Retriever(index=index).search("tokens", top_k=3)
        assert results
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        assert results[0].score >= results[-1].score

    async def test_stamps_the_retriever_name(self) -> None:
        index = FakeLexicalIndex()
        populate(index)
        results = await BM25Retriever(index=index).search("tokens", top_k=1)
        assert results[0].retriever == "bm25"

    async def test_top_k_limits_results(self) -> None:
        index = FakeLexicalIndex()
        populate(index)
        assert len(await BM25Retriever(index=index).search("tokens", top_k=1)) == 1

    async def test_non_positive_top_k_short_circuits(self) -> None:
        index = FakeLexicalIndex()
        populate(index)
        assert await BM25Retriever(index=index).search("tokens", top_k=0) == []
        assert index.queries == []

    async def test_filters_reach_the_index(self) -> None:
        index = FakeLexicalIndex()
        populate(index)
        results = await BM25Retriever(index=index).search(
            "tokens", top_k=10, filters=SearchFilters(source_types=[SourceType.PDF])
        )
        assert results
        assert all(r.source_type is SourceType.PDF for r in results)

    async def test_records_no_embedding_time(self) -> None:
        """BM25 has no model in the path; a comparison must see that as zero."""
        index = FakeLexicalIndex()
        populate(index)
        response = await SearchService(retriever=BM25Retriever(index=index)).search(
            "tokens", top_k=2
        )
        assert response.retrieval_strategy == "bm25"
        assert response.timing.embedding_ms == 0.0
        assert response.timing.retrieval_ms > 0


def compiled(query: str = "how are bearer tokens verified", **kwargs: object) -> str:
    index = PostgresBM25Index(None)  # type: ignore[arg-type]
    statement = index._build_statement(query, top_k=10, filters=kwargs.get("filters"))  # type: ignore[arg-type]
    return str(
        statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )


class TestGeneratedSQL:
    def test_scores_with_bm25_not_ts_rank(self) -> None:
        """Guards the project's headline honesty claim about this component."""
        sql = compiled().lower()
        assert "ts_rank" not in sql
        assert "ln(" in sql  # the IDF term

    def test_uses_or_semantics_for_candidates(self) -> None:
        assert "' | '" in compiled()

    def test_does_not_use_to_tsquery(self) -> None:
        """``to_tsquery`` re-parses quoted tokens and shatters URL/path lexemes."""
        assert "to_tsquery" not in compiled()
        assert "as tsquery" in compiled().lower()

    def test_query_and_document_share_one_analyzer(self) -> None:
        """Different text search configs would silently stop matching."""
        assert compiled().lower().count("cast('english' as regconfig)") == 1

    def test_filters_are_pushed_into_sql(self) -> None:
        sql = compiled(filters=SearchFilters(source_types=[SourceType.PDF]))
        # Once for the candidate set, once for the collection statistics: the
        # corpus a filtered search is scored against must be the filtered one.
        assert sql.count("documents.source_type IN ('pdf')") == 2

    def test_ordering_is_deterministic(self) -> None:
        assert "ORDER BY scored.score DESC, scored.chunk_id" in compiled()

    def test_k1_and_b_reach_the_formula(self) -> None:
        index = PostgresBM25Index(None, k1=2.0, b=0.0)  # type: ignore[arg-type]
        sql = str(
            index._build_statement("tokens", top_k=5, filters=None).compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            )
        )
        assert "3.0" in sql  # k1 + 1
        assert "0.0 * " in sql or "0.0*" in sql  # b, disabling length normalisation

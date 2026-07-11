"""``recall`` — the command line interface.

The CLI talks to ``recall.pipeline`` directly. It never requires the API server
to be running, which keeps the core usable as a library and makes experiments
reproducible from a shell.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from recall import __version__
from recall.cli.console import console, error_console, fail
from recall.cli.templates import ENV_EXAMPLE, EXAMPLE_DOC, GITIGNORE, RECALL_YAML
from recall.config.settings import Settings, find_config_file, load_settings
from recall.connectors import connector_registry, create_connector
from recall.core.chunking import chunker_registry
from recall.core.embeddings import embedder_registry
from recall.core.errors import RecallError
from recall.core.models import SearchFilters, SourceType, SyncResult
from recall.core.reranking import reranker_registry
from recall.core.retrieval import fusion_registry, retriever_registry
from recall.observability.logging import configure_logging
from recall.pipeline.factory import RecallContext, build_context

app = typer.Typer(
    name="recall",
    help="An open-source framework for building, evaluating and experimenting "
    "with knowledge retrieval systems.",
    no_args_is_help=True,
    add_completion=False,
)
documents_app = typer.Typer(help="Inspect ingested documents.", no_args_is_help=True)
app.add_typer(documents_app, name="documents")


# --- shared helpers ---------------------------------------------------------


def _settings(config: Path | None = None) -> Settings:
    try:
        settings = load_settings(config)
    except RecallError as exc:
        fail(str(exc), hint="Run `recall init` to create a valid recall.yaml.")
    configure_logging(settings.logging.level, settings.logging.format)
    return settings


def _context(settings: Settings, *, retrieval_strategy: str | None = None) -> RecallContext:
    try:
        return build_context(settings, retrieval_strategy=retrieval_strategy)
    except RecallError as exc:
        fail(str(exc))


def _print_sync_result(result: SyncResult, *, label: str) -> None:
    from rich.table import Table

    table = Table(title=f"{label} sync", title_justify="left", header_style="bold")
    table.add_column("outcome")
    table.add_column("count", justify="right")
    for name, value in (
        ("created", result.created),
        ("updated", result.updated),
        ("unchanged", result.unchanged),
        ("deleted", result.deleted),
        ("failed", result.failed),
    ):
        table.add_row(name, str(value))
    table.add_row("chunks written", str(result.chunks_written), style="bold")
    console.print(table)

    for item in result.items:
        if item.error:
            error_console.print(f"[red]failed[/red] {item.source_id}: {item.error}")


# --- commands ---------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the Recall version."""
    console.print(__version__)


@app.command()
def init(
    directory: Annotated[Path, typer.Argument(help="Project directory.")] = Path("."),
    provider: Annotated[
        str,
        typer.Option(
            "--embedding-provider",
            help="Embedding provider to write into recall.yaml.",
        ),
    ] = "sentence_transformers",
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
) -> None:
    """Create recall.yaml, .env.example and an example corpus."""
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    if provider not in embedder_registry:
        fail(
            f"Unknown embedding provider {provider!r}.",
            hint=f"Available: {', '.join(embedder_registry.names())}",
        )

    presets = {
        "sentence_transformers": ("BAAI/bge-base-en-v1.5", 768),
        "openai": ("text-embedding-3-small", 1536),
        "hash": ("hash-v1", 384),
    }
    model, dimensions = presets.get(provider, ("hash-v1", 384))

    files: dict[Path, str] = {
        directory / "recall.yaml": RECALL_YAML.replace("__PROJECT_NAME__", directory.name)
        .replace("__EMBEDDING_PROVIDER__", provider)
        .replace("__EMBEDDING_MODEL__", model)
        .replace("__EMBEDDING_DIMENSIONS__", str(dimensions)),
        directory / ".env.example": ENV_EXAMPLE,
        directory / ".gitignore": GITIGNORE,
        directory / "examples" / "documents" / "authentication.md": EXAMPLE_DOC,
    }

    written, skipped = [], []
    for path, content in files.items():
        if path.exists() and not force:
            skipped.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)

    for path in written:
        console.print(f"[green]created[/green] {path.relative_to(directory)}")
    for path in skipped:
        console.print(f"[yellow]exists[/yellow]  {path.relative_to(directory)} (use --force)")

    console.print(
        "\nNext:\n"
        "  1. [bold]docker compose up -d postgres[/bold]\n"
        "  2. [bold]recall migrate[/bold]\n"
        "  3. [bold]recall ingest ./examples/documents[/bold]\n"
        '  4. [bold]recall search "How does authentication work?"[/bold]'
    )


@app.command()
def migrate(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    revision: Annotated[str, typer.Option(help="Target revision.")] = "head",
) -> None:
    """Create or update the database schema."""
    settings = _settings(config)
    from recall.storage.postgres import migrate as migrations

    try:
        migrations.upgrade(settings, revision)
    except Exception as exc:
        fail(
            f"Migration failed: {exc}",
            hint="Is PostgreSQL running and does the database exist? "
            "Try `docker compose up -d postgres`.",
        )
    console.print(f"[green]schema up to date[/green] ({revision})")


@app.command()
def status(config: Annotated[Path | None, typer.Option("--config", "-c")] = None) -> None:
    """Show configuration and database health."""
    settings = _settings(config)
    config_path = config or find_config_file()

    from rich.table import Table

    table = Table(title="recall status", title_justify="left", header_style="bold")
    table.add_column("setting")
    table.add_column("value")
    table.add_row("version", __version__)
    table.add_row("config file", str(config_path) if config_path else "(defaults)")
    table.add_row("embedding", f"{settings.embedding.provider}:{settings.embedding.model}")
    table.add_row("dimensions", str(settings.embedding.dimensions))
    table.add_row(
        "chunking",
        f"{settings.chunking.strategy} "
        f"(size={settings.chunking.chunk_size}, overlap={settings.chunking.overlap})",
    )
    table.add_row("retrieval", settings.retrieval.default)

    async def check() -> dict[str, Any]:
        context = _context(settings)
        try:
            return await context.storage.health()
        finally:
            await context.close()

    try:
        health = asyncio.run(check())
    except Exception as exc:
        table.add_row("database", f"[red]unreachable[/red] ({type(exc).__name__})")
        console.print(table)
        raise typer.Exit(1) from None

    table.add_row("database", "[green]connected[/green]")
    table.add_row("pgvector", str(health.get("pgvector_version") or "[red]missing[/red]"))
    table.add_row("documents", str(health.get("documents")))
    table.add_row("chunks", str(health.get("chunks")))
    table.add_row("vectors", str(health.get("vectors")))
    console.print(table)


@app.command()
def connectors() -> None:
    """List registered components."""
    from rich.table import Table

    table = Table(title="registered components", title_justify="left", header_style="bold")
    table.add_column("kind")
    table.add_column("names")
    table.add_row("connectors", ", ".join(connector_registry.names()))
    table.add_row("chunkers", ", ".join(chunker_registry.names()))
    table.add_row("embedders", ", ".join(embedder_registry.names()))
    table.add_row("retrievers", ", ".join(retriever_registry.names()))
    table.add_row("fusion", ", ".join(fusion_registry.names()))
    table.add_row("rerankers", ", ".join(reranker_registry.names()))
    console.print(table)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory to ingest.")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    source: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Connector to use: auto | filesystem | pdf.",
        ),
    ] = "auto",
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-chunk and re-embed even when unchanged."),
    ] = False,
    prune: Annotated[
        bool,
        typer.Option("--prune/--no-prune", help="Delete documents missing from the source."),
    ] = True,
    recursive: Annotated[bool, typer.Option("--recursive/--no-recursive")] = True,
) -> None:
    """Ingest local files and PDFs into the knowledge base."""
    path = path.expanduser().resolve()
    if not path.exists():
        fail(f"{path} does not exist")

    settings = _settings(config)

    if source == "auto":
        kinds = ["filesystem", "pdf"]
    elif source in connector_registry:
        kinds = [source]
    else:
        fail(
            f"Unknown connector {source!r}.",
            hint=f"Available: auto, {', '.join(connector_registry.names())}",
        )

    async def run() -> int:
        context = _context(settings)
        failures = 0
        try:
            pipeline = context.ingestion
            for kind in kinds:
                connector = create_connector(kind, root=path, recursive=recursive)
                discovered = await connector.discover()
                if not discovered:
                    if source != "auto":
                        console.print(f"[yellow]no matching files for connector {kind}[/yellow]")
                    continue
                result = await pipeline.sync(connector, force=force, prune=prune)
                _print_sync_result(result, label=kind)
                failures += result.failed
        finally:
            await context.close()
        return failures

    try:
        failed = asyncio.run(run())
    except RecallError as exc:
        fail(str(exc))
    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}", hint="Run `recall status` to check the database.")

    if failed:
        raise typer.Exit(1)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="The search query.")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1, max=200)] = 10,
    strategy: Annotated[
        str | None,
        typer.Option(
            "--strategy",
            "-s",
            help="Retrieval strategy. Defaults to retrieval.default in recall.yaml.",
        ),
    ] = None,
    rerank: Annotated[
        str | None,
        typer.Option(
            "--rerank",
            help="Reranker to apply. 'off' disables it. Defaults to reranking.* in recall.yaml.",
        ),
    ] = None,
    source_type: Annotated[
        list[str] | None, typer.Option("--source-type", help="Filter by source type.")
    ] = None,
    file_type: Annotated[
        list[str] | None, typer.Option("--file-type", help="Filter by file extension.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table.")] = False,
    show_content: Annotated[int, typer.Option("--chars", help="Content preview length.")] = 220,
) -> None:
    """Search the knowledge base."""
    settings = _settings(config)

    try:
        filters = SearchFilters(
            source_types=[SourceType(value) for value in (source_type or [])],
            file_types=[value.lstrip(".").lower() for value in (file_type or [])],
        )
    except ValueError as exc:
        fail(
            f"Invalid filter: {exc}",
            hint=f"Valid source types: {', '.join(t.value for t in SourceType)}",
        )

    if strategy is not None and strategy not in retriever_registry:
        fail(
            f"Unknown retrieval strategy {strategy!r}.",
            hint=f"Available: {', '.join(retriever_registry.names())}",
        )

    if rerank is not None:
        if rerank == "off":
            settings = settings.model_copy(
                update={"reranking": settings.reranking.model_copy(update={"enabled": False})}
            )
        elif rerank in reranker_registry:
            settings = settings.model_copy(
                update={
                    "reranking": settings.reranking.model_copy(
                        update={"enabled": True, "strategy": rerank}
                    )
                }
            )
        else:
            fail(
                f"Unknown reranker {rerank!r}.",
                hint=f"Available: off, {', '.join(reranker_registry.names())}",
            )

    async def run() -> Any:
        context = _context(settings, retrieval_strategy=strategy)
        try:
            return await context.search.search(query, top_k=top_k, filters=filters)
        finally:
            await context.close()

    try:
        response = asyncio.run(run())
    except RecallError as exc:
        fail(str(exc))
    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}", hint="Run `recall status` to check the database.")

    if as_json:
        console.print_json(json.dumps(response.model_dump(mode="json")))
        return

    if not response.results:
        console.print("[yellow]no results[/yellow]")
        console.print(
            f"[dim]{response.timing.total_ms:.1f} ms "
            f"(embedding {response.timing.embedding_ms:.1f} ms, "
            f"retrieval {response.timing.retrieval_ms:.1f} ms)[/dim]"
        )
        return

    from rich.table import Table

    table = Table(
        title=f'search: "{query}"  [{response.retrieval_strategy}]',
        title_justify="left",
        header_style="bold",
        show_lines=True,
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("score", justify="right", width=7)
    table.add_column("document")
    table.add_column("chunk")

    for result in response.results:
        preview = " ".join(result.content.split())
        if len(preview) > show_content:
            preview = preview[: show_content - 1] + "…"
        source = result.source_type.value if result.source_type else "?"
        table.add_row(
            str(result.rank),
            f"{result.score:.4f}",
            f"{result.document_title or '(untitled)'}\n[dim]{source} · "
            f"{result.metadata.get('filename', '')}[/dim]",
            preview,
        )

    console.print(table)
    stages = [
        f"embedding {response.timing.embedding_ms:.1f} ms",
        f"retrieval {response.timing.retrieval_ms:.1f} ms",
    ]
    if response.timing.fusion_ms:
        stages.append(f"fusion {response.timing.fusion_ms:.1f} ms")
    if response.reranked:
        stages.append(
            f"reranking {response.timing.reranking_ms:.1f} ms over {response.candidates} candidates"
        )
    console.print(
        f"[dim]{len(response.results)} results in {response.timing.total_ms:.1f} ms "
        f"({', '.join(stages)}) · request {response.request_id}[/dim]"
    )


@documents_app.command("list")
def documents_list(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=1000)] = 20,
    offset: Annotated[int, typer.Option("--offset", min=0)] = 0,
    source_type: Annotated[list[str] | None, typer.Option("--source-type")] = None,
) -> None:
    """List ingested documents."""
    settings = _settings(config)
    types = [SourceType(value) for value in (source_type or [])] or None

    async def run() -> tuple[list[Any], int]:
        context = _context(settings)
        try:
            docs = await context.storage.documents.list(
                source_types=types, limit=limit, offset=offset
            )
            total = await context.storage.documents.count(source_types=types)
            return docs, total
        finally:
            await context.close()

    try:
        docs, total = asyncio.run(run())
    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}", hint="Run `recall status` to check the database.")

    from rich.table import Table

    table = Table(title=f"documents ({total} total)", title_justify="left", header_style="bold")
    table.add_column("title")
    table.add_column("source")
    table.add_column("source id")
    table.add_column("chars", justify="right")
    table.add_column("updated")
    for document in docs:
        table.add_row(
            document.title,
            document.source_type.value,
            document.source_id,
            str(len(document.content)),
            document.updated_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


@documents_app.command("show")
def documents_show(
    document_id: Annotated[str, typer.Argument(help="Document UUID.")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    chunks: Annotated[
        bool, typer.Option("--chunks", help="Also list the document's chunks.")
    ] = False,
) -> None:
    """Show one document and, optionally, its chunks."""
    import uuid as _uuid

    settings = _settings(config)
    try:
        parsed = _uuid.UUID(document_id)
    except ValueError:
        fail(f"{document_id!r} is not a valid UUID")

    async def run() -> tuple[Any, list[Any]]:
        context = _context(settings)
        try:
            document = await context.storage.documents.get(parsed)
            rows = await context.storage.chunks.list_for_document(parsed) if chunks else []
            return document, rows
        finally:
            await context.close()

    try:
        document, rows = asyncio.run(run())
    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}")

    if document is None:
        fail(f"No document with id {document_id}")

    console.print(f"[bold]{document.title}[/bold]")
    console.print(f"[dim]{document.uri}[/dim]")
    console.print(
        f"source={document.source_type.value} source_id={document.source_id} "
        f"checksum={document.checksum[:12]}…"
    )
    console.print_json(json.dumps(document.metadata, default=str))

    if chunks:
        from rich.table import Table

        table = Table(title=f"{len(rows)} chunks", title_justify="left", header_style="bold")
        table.add_column("#", justify="right", width=3)
        table.add_column("tokens", justify="right")
        table.add_column("content")
        for chunk in rows:
            preview = " ".join(chunk.content.split())
            table.add_row(
                str(chunk.position),
                str(chunk.token_count),
                preview[:160] + ("…" if len(preview) > 160 else ""),
            )
        console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()

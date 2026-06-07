# Ingestion and incremental sync

Re-embedding an unchanged corpus is the most common way a retrieval system wastes money and time. Recall treats "do nothing" as the expected outcome of a repeat ingest.

## The algorithm

```
discover()
    |
    +-- did discovery give us a checksum, and does it match what is stored?
    |        yes -> UNCHANGED, no fetch at all
    |        no  -> continue
    |
fetch(item) -> Document  (normalized content, checksum computed from it)
    |
    +-- does the document checksum match what is stored?
    |        yes -> UNCHANGED, no chunking, no embedding
    |        no  -> continue
    |
chunk -> embed -> index   (single transaction)
    |
prune: delete stored documents of this source_type that discovery did not return
```

## Why two comparisons

The two passes answer different questions.

**Pass 1** uses whatever the connector learned cheaply during discovery. A GitHub connector gets a blob SHA from the tree listing; a Notion connector gets `last_edited_time`. When that is available and matches, Recall skips the network fetch entirely.

**Pass 2** is authoritative and always runs when pass 1 could not decide. It compares a SHA-256 over the document's *normalized title and content*. This is what makes the following cases correct:

- A file is touched or re-saved with identical bytes: `mtime` changes, checksum does not → unchanged.
- A file is reformatted in a way normalization erases (line endings, trailing whitespace): checksum does not change → unchanged.
- A document is retitled but its body is identical: checksum *does* change → reindexed, because the title is part of what gets retrieved.

The filesystem connector deliberately does **not** advertise a checksum during discovery — computing it would mean reading the file, which is the fetch. It relies on pass 2.

## Deletions

`prune` (on by default) deletes stored documents whose `source_id` no longer appears in `discover()`. This is scoped to the connector's `source_type`, so ingesting a directory of Markdown never deletes PDFs from the same directory, and two connectors can share a root safely.

Deletion cascades: `documents` → `chunks` → `chunk_embeddings`. An orphaned vector — one that outlives the chunk it described — would silently corrupt every future search, so the foreign keys enforce it rather than the application.

## Atomicity

Embedding happens *before* the transaction opens. The transaction then does three things and either all of them land or none do:

```python
async with session_scope(self.sessions) as session:
    await upsert_document(session, document)
    written = await replace_chunks(session, document.id, chunks)
    await upsert_embeddings(session, rows)
```

A failed embedding call therefore leaves the previous state completely intact — the old document, its old chunks, and its old vectors. This is the guarantee tested by `test_a_bad_vector_leaves_no_partial_state`.

## Failure isolation

One unreadable file must not abort a 10,000-document ingest. Each item is processed independently; `RecallError` and unexpected exceptions alike are caught, recorded as a `FAILED` `SyncItemResult` with the error message, and the sync continues. `TransientError` is retried with exponential backoff and jitter before being recorded as a failure.

The CLI exits non-zero when any item failed, so a scripted ingest still fails loudly.

## Forcing a rebuild

`recall ingest ./docs --force` skips both checksum comparisons. Use it after changing the chunking strategy or the embedding model — neither of those changes the source content, so incremental sync would otherwise correctly decide there is nothing to do.

Document and chunk IDs are deterministic, so a forced rebuild reuses the same primary keys rather than accumulating duplicates.

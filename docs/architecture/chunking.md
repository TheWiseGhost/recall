# Chunking

A chunker turns one `Document` into an ordered list of `Chunk`s. It is the least glamorous stage in the pipeline and one of the most consequential: a chunk is the unit that gets embedded, indexed, retrieved and shown, so a bad boundary is an error every later stage inherits and none can repair.

Which strategy wins is corpus-dependent, which is why there are four and why they are selected by name.

| Strategy | Cuts on | Embeds at ingest | Notes |
|---|---|---|---|
| `fixed` | token budget | chunks only | the baseline everything is measured against |
| `sentence` | sentence boundaries, packed to a budget | chunks only | never severs a sentence |
| `semantic` | embedding distance between sentences | **every sentence, plus chunks** | boundaries follow topic |
| `hierarchical` | token budget, at two levels | both levels | emits parent/child structure |

```yaml
chunking:
  strategy: fixed
  chunk_size: 512
  overlap: 64                # fixed, hierarchical
  overlap_sentences: 1       # sentence
  breakpoint_percentile: 0.95  # semantic
  buffer_size: 1             # semantic
  max_chunk_size: 1024       # semantic
  parent_chunk_size: 2048    # hierarchical
```

## The protocol is async

```python
class Chunker(Protocol):
    name: str

    async def chunk(self, document: Document) -> list[Chunk]: ...
```

Chunking is not always pure text manipulation — semantic chunking has to embed candidate sentences to find its boundaries, and an LLM-guided chunker would call out too. Strategies that need neither simply never await.

Implementers still write one synchronous method. `ChunkerBase` handles ID derivation, token counting and metadata propagation; a subclass implements `split(document) -> [(text, start, end)]`. Only when splitting needs I/O do you override `split_async` instead.

## `fixed`

Sliding window of `chunk_size` tokens advancing by `chunk_size - overlap`, cutting on token boundaries in the original text so a chunk never splits a word and character offsets always map back into the source.

The baseline. Every other strategy has to earn its complexity against this one.

## `sentence`

Packs whole sentences into windows of at most `chunk_size` tokens. Fixed-size chunking routinely severs a sentence, and half a sentence embeds to something that means neither half.

**Overlap is counted in sentences, not tokens.** "Carry one sentence of context forward" is a statement someone can reason about; "carry 64 tokens" would cut mid-sentence again and undo the entire point.

**A sentence longer than `chunk_size` becomes its own oversized chunk.** Splitting it would reintroduce the problem the strategy exists to solve. The oversize is visible in `token_count`; silently truncating it would not be.

### Sentence segmentation is a heuristic

`core/chunking/sentences.py` is regex-based and dependency-free. Recall does not pull in NLTK or spaCy for it: both are large, both want a download step, and a missed boundary here produces a slightly differently-shaped chunk rather than a wrong answer.

It handles an explicit abbreviation list (`Dr.`, `e.g.`, `Fig.`, …), initials, decimals, closing quotes, and treats a blank line as a hard boundary — headings and list items rarely end in a period. It does not handle abbreviations outside that list, `etc.` at a genuine sentence end, or languages that do not delimit with `.?!`. Pass a different splitter if your corpus needs one; that is why it is a function.

## `semantic`

Embeds each sentence, measures cosine distance between consecutive sentences, and breaks where that distance spikes. A chunk then covers one topic rather than one token count.

**The threshold is a percentile of the distances in that document, not an absolute distance.** Absolute cosine distances are not comparable across embedding models or across documents of different styles, so a fixed threshold would silently mean something different for every model swapped in.

**`buffer_size` widens what gets embedded.** With `buffer_size=1` each sentence is embedded together with its neighbours. A lone sentence often embeds to something noisy — a pronoun, a bare clause — and comparing two noisy vectors produces spurious breakpoints.

**`max_chunk_size` is a backstop.** A long stretch of on-topic prose would otherwise become one enormous chunk no retriever can rank usefully. When a group exceeds it, the cut lands at the largest remaining distance *inside* the group, so it still falls on a topic seam rather than an arbitrary token offset.

**It uses the pipeline's embedder.** Boundaries that depended on a model nothing else in the system knew about would be unreproducible, so `build_chunker` passes the configured embedder in, and the model key is recorded in every chunk's `chunker_params`.

**It costs.** Every sentence in the corpus is embedded at ingest time, on top of the chunks themselves — roughly double on a paid API. Whether that buys retrieval quality is experiment 001, not an assumption.

## `hierarchical`

Retrieval wants small chunks; answering wants large ones. This emits both: small children are the retrieval units, and each carries `parent_id` pointing at the larger span it came from.

The expansion step — retrieve a child, return its parent — is context selection and lands in Milestone 4. What this chunker does is record the structure. `chunks.parent_id` has existed since the initial migration for exactly this.

**Positions are unique across both levels, children first in reading order.** Children are the retrieval units, so their positions must be their reading order — neighbour expansion and offset arithmetic depend on it. Uniqueness across levels is not cosmetic: a chunk's ID is `uuid5(document_id, position, content_checksum)`, and a parent with exactly one child can have byte-identical text, so a reused position would collide their IDs.

**Parents are returned first even though they are positioned last.** Storage inserts in list order and `parent_id` is a self-referencing foreign key: a row cannot point at one that has not been written yet.

**Parents do not overlap.** Overlapping parents would assign the same text to two hierarchies and double-count it on expansion. Children overlap normally.

**Both levels are embedded**, because the pipeline embeds every chunk it is given. That makes both granularities searchable, which is the point of the comparison — and it costs. **TODO / FUTURE:** embed children only, once parent-child context selection exists to make parents retrievable indirectly.

## Provenance

Every chunk records `chunker` and `chunker_params` in its metadata, so a result file says which strategy and which parameters produced the chunk it is reporting on. Source document metadata is copied onto the chunk too, so metadata filtering works against chunks without a join back to the document.

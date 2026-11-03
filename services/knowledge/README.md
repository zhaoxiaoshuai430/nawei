# Knowledge GraphRAG Service

The Knowledge service owns public document ingestion and retrieval. It does not
store user conversation memory.

## Storage

- PostgreSQL: document status and ingestion metadata
- Redis: retrieval cache
- Qdrant: chunk text, payload filters, and embeddings
- Neo4j: documents, chunks, entities, mentions, and semantic relations

The same UUID is used as the Qdrant point ID and Neo4j `Chunk.id`, so vector
results can seed graph traversal without an additional mapping database.

## Local APIs

| API | Purpose |
|---|---|
| `GET /health` | PostgreSQL, Redis, Qdrant, and Neo4j readiness |
| `POST /api/v1/knowledge/documents/text` | Index JSON text content |
| `POST /api/v1/knowledge/documents/structured` | Index citation-aware chunks and explicit graph relations |
| `POST /api/v1/knowledge/documents/upload` | Upload TXT or Markdown |
| `POST /api/v1/knowledge/retrieve` | Graph candidate resolution, filtered vector recall, and bounded graph expansion |
| `GET /api/v1/knowledge/documents/{id}` | Inspect indexing status |
| `DELETE /api/v1/knowledge/documents/{id}` | Delete document data from both stores |

Structured chunks may include `source_name`, `source_url`, and
`source_locator`. Retrieval returns those fields so the LLM can cite an exact
workbook sheet/row or an original reference URL.

## Concept-grounded retrieval

LLM TutorGraph requests may include `anchor_terms` such as `三角函数`, `正弦`,
and `余弦`. When anchors are present, Knowledge keeps only chunks that actually
mention at least one anchor before expanding Neo4j relations. This prevents a
broad discipline node such as `函数论` from being cited as evidence for a
specific formula that is not present in the source chunk.

TutorGraph may also send ordered `graph_candidates`. Knowledge resolves those
candidates against Neo4j before Qdrant retrieval, applies a `subject`, `domain`,
or `concept` payload filter when possible, falls back to broad vector retrieval
when the filtered search is empty, and expands only whitelisted relation groups.
The response exposes `resolved_entities`, `relation_groups`,
`graph_match_level`, and `grounding_status` for runtime inspection.

The first curated concept pack is stored at
`knowledge_data/math/trigonometry.json`. Import or validate it with:

```powershell
.\.venv\Scripts\python.exe scripts\import_structured_knowledge.py --dry-run
.\.venv\Scripts\python.exe scripts\import_structured_knowledge.py
```

Neo4j stores the concept structure and prerequisite relationships. Qdrant
stores definitions, formulas, examples, source locators, and teaching text.
The trigonometry pack reuses the Atlas `数学 [KnowledgeNode]` entity and links
`数学 → 初等数学 → 三角学 → 三角函数` instead of creating a separate Subject node.

## Human-readable citation paths

Atlas imports keep workbook sheet and row numbers only in internal chunk
metadata for auditing. User-facing citation fields are normalized as follows:

- taxonomy nodes: classification authority + full knowledge path + exact standard locator;
- curriculum and school evidence: publishing institution/title + country/age/course path + public URL;
- reference catalog entries: publisher + publication title + chapter/page + public URL;
- derived matrices and findings: named project analysis + country/age/domain path, without a fabricated URL.

Prompt character budgets preserve each included chunk's complete attribution.
Long evidence is excerpted with an explicit truncation marker; chunks whose
attribution cannot fit are omitted, and source numbers retain their original
positions. Graph facts are included whole. Citation instructions require actual
support for the answer: taxonomy entries support classification, not technical
claims, and missing source fields must not be invented.

Independent retrieval branches still run concurrently. If entity resolution,
embedding, or graph expansion fails, sibling tasks are cancelled and awaited
before the error returns. This also applies to individual Neo4j relation groups;
failed retrievals do not cache partial results.

The importer dry-run reports `citation_audit`. Both
`internal_source_names` and `internal_source_locators` must be zero before an
Atlas migration is applied.

## Human Knowledge Atlas import

Use the Excel workbook as the authoritative source. The generated HTML repeats
the same facts, so the importer validates its mirrored counts and indexes only
one HTML overview chunk to avoid duplicate retrieval results.

```powershell
.\.venv\Scripts\python.exe scripts\import_knowledge_atlas.py `
  --xlsx "D:\daima\人类知识图谱总表0824(1).xlsx" `
  --html "D:\daima\Human_Knowledge_Education_Atlas_完整统一版_v2.html" `
  --knowledge-base-id default `
  --dry-run
```

Remove `--dry-run` after the Knowledge service has been restarted with the
structured endpoint. The importer creates eight workbook documents plus one
HTML overview document. Every workbook row retains its sheet name, Excel row,
record ID, and any source URL present in the workbook. Importing into `default`
keeps the data available to the existing LLM and voice conversation flow.

The development default uses deterministic hash embeddings and rule extraction
only to validate infrastructure. Real semantic retrieval requires a local
FastEmbed model or a configured embedding endpoint. LLM graph extraction is a
separate option, controlled by `KNOWLEDGE_GRAPH_EXTRACTOR=llm`.

## Local Chinese semantic embeddings

Install the optional CPU dependency in the main virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r deploy/requirements-embeddings.txt
.\.venv\Scripts\python.exe -c "from fastembed import TextEmbedding; model = TextEmbedding(model_name='BAAI/bge-small-zh-v1.5', cache_dir='.runtime/embedding-cache', threads=2, providers=['CPUExecutionProvider']); print(len(next(model.embed(['test']))))"
```

The preparation command downloads the model and must print `512`. The service
itself uses `local_files_only=True`: download the model before starting it.
It warms the CPU model before reporting readiness and does not use the TTS GPU.
Inference runs off the event loop with at most one batch per provider instance.

Build a separate index while ingestion/deletion is paused. From the repo root:

```powershell
. .\scripts\project_runtime.ps1
Initialize-NaweiLocalEnvironment
.\.venv\Scripts\python.exe scripts/reindex_semantic.py --source knowledge_chunks_v1 --inspect
.\.venv\Scripts\python.exe scripts/reindex_semantic.py --source knowledge_chunks_v1 --target knowledge_chunks_bge_zh_20260907
.\.venv\Scripts\python.exe scripts/evaluate_semantic_pipeline.py --source knowledge_chunks_v1 --target knowledge_chunks_bge_zh_20260907
```

Migration accepts ready `hash-embedding-v1` payloads only and refuses an existing
target. It keeps the original collection and chunk IDs, so citations and Neo4j
links remain valid. The evaluation script compares existing indexes using the
real hybrid retrieval pipeline. Its small concept-term fixture reports Hit@5
and MRR@5, not an exhaustive relevance or unanswerable-query benchmark.

After reviewing the reports, set the local `.env.knowledge` to:

```dotenv
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DIMENSION=512
EMBEDDING_THREADS=2
QDRANT_COLLECTION=knowledge_chunks_bge_zh_20260907
```

Restart with `scripts/restart_robot_services.ps1 -Names knowledge`; verify
`GET http://127.0.0.1:8003/health` reports `ready: true` and the BGE model.
Configuration precedence is process environment, then `.env.knowledge`, then
`.env`, then code defaults. Explicit process variables can override this file.
`EMBEDDING_CACHE_DIR` defaults to the repo's `.runtime/embedding-cache`.

Existing collections must match the configured dimension, cosine distance, and
every point's embedding version. Incompatible collections are rejected, never
silently recreated. Retrieval cache keys include the collection, provider,
model, and dimension. `metadata.timings_ms` separates cache lookup, entity
resolution, embedding, vector/lexical search, and graph expansion.

`.env.knowledge.example` contains the original hash configuration for rollback.
Restore those values and restart Knowledge only when the old collection still
matches the shared catalog and graph. It is a migration-time snapshot, not a
live replica: new ingestion or deletions after activation require reconciliation
before rollback. Keep writes paused across migration and activation.

Production Compose builds the `knowledge` target of `deploy/Dockerfile.app`,
which installs `deploy/requirements-embeddings.txt`. It mounts the prepared
`models/embeddings` directory read-only at `/models/embeddings` and passes
model/index/cache settings explicitly. The generic `app` target does not install
FastEmbed. The ignored local `.env.knowledge` file is not copied into either image.

Graph relations are validated structurally and written during successful document
ingestion, using either rule or LLM extraction, or explicit structured input.
There is no implemented human approval queue between extraction and publication;
source review remains an operational responsibility. Ordinary chat answers are
not automatically ingested as knowledge documents.

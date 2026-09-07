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
only to validate infrastructure. Production must configure a semantic embedding
endpoint and should set `KNOWLEDGE_GRAPH_EXTRACTOR=llm`.

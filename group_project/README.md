# Group Project - Search Engine / RAG Chatbot

## Goal

Build a RAG chatbot for questions about Vietnamese drug prevention law and
related news. The group project is split into four modules, all connected by
the shared contract in `group_project/system_contracts.py`.

## Members

| Member | Student ID | Module | Path |
|--------|------------|--------|------|
| Le Dam Quan | 2A202600930 | Chat UI / UX | `src/module_chat_ui/` |
| Nguyen Tien Dat | 2A202600595 | RAG Core | `src/module_rag_core/` |
| Tran Nguyen Dang Khoa | 2A202600922 | Dataset Creator | `src/module_dataset_creator/` |
| Tran Hoang Nam | 2A202600870 | Evaluation | `src/module_evaluation/` |

## Architecture

```text
Streamlit UI
  -> RAGCoreInterface
  -> RAG Core
       -> retrieval / reranking / generation
  -> RAGAnswer(answer, sources, standalone_query)

Golden Dataset
  -> Evaluation Pipeline
       -> config A: hybrid + reranker
       -> config B: question-only + no reranker
  -> results.md + REPORT.md
```

## Module Layout

```text
group_project/
|-- app.py
|-- system_contracts.py
|-- README.md
|-- REPORT.md
|-- src/
|   |-- module_chat_ui/
|   |-- module_dataset_creator/
|   |-- module_evaluation/
|   `-- module_rag_core/
`-- tests/
```

## Integration Contract

`system_contracts.py` defines:

- `RAGConfig`
- `Document`
- `ChatMessage`
- `RAGAnswer`
- `RAGCoreInterface`

The UI and evaluation modules call only `RAGCoreInterface`. This keeps each
member's module replaceable without rewriting the rest of the app.

## Run Chat UI

From the repository root:

```bash
streamlit run group_project/app.py
```

The UI tries to load `src.module_rag_core.rag_engine.RAGCoreEngine`. If Gemini,
Weaviate, or API keys are unavailable, the integrated RAG core falls back to
offline demo documents so the chat flow still works for local testing.

## Run Dataset Validation

```bash
python group_project/src/module_dataset_creator/validate_schema.py
```

## Run Evaluation

```bash
python group_project/src/module_evaluation/eval_pipeline.py
```

Outputs:

- `group_project/src/module_evaluation/results.md`
- `group_project/REPORT.md`

## Import Data To Weaviate

After `data/standardized/legal` and `data/standardized/news` are available,
load them into the RAG core vector store with:

```bash
python group_project/import_data.py
```

The script reads `GEMINI_API_KEY`, `WEAVIATE_URL`, and `WEAVIATE_API_KEY` from
`.env`. If Gemini is not configured, it uses zero vectors so the import flow can
still be tested.

## Run Integration Test

```bash
pytest group_project/tests/test_system_integration.py -v
```

## Current Status

- Dataset creator has a 15+ item golden dataset and schema validator.
- RAG core is integrated from Nguyen Tien Dat's branch with offline-safe
  Gemini and Weaviate fallbacks plus a Weaviate import script.
- Chat UI loads the shared RAG core through `RAGCoreInterface`.
- Evaluation has 4 metrics, 2 A/B configs, and worst-performer reporting.

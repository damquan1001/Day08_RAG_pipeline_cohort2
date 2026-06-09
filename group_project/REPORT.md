# Group Project Report

Date: 2026-06-08

## Scope

This report documents the integrated group RAG chatbot pipeline, the module
owners, evaluation results, UI smoke checks, and test evidence.

## Team Modules

| Member | Student ID | Module | Integrated path |
|--------|------------|--------|-----------------|
| Le Dam Quan | 2A202600930 | Chat UI / UX | `src/module_chat_ui/` |
| Nguyen Tien Dat | 2A202600595 | RAG Core | `src/module_rag_core/` |
| Tran Nguyen Dang Khoa | 2A202600922 | Dataset Creator | `src/module_dataset_creator/` |
| Tran Hoang Nam | 2A202600870 | Evaluation | `src/module_evaluation/` |

## Integration Summary

- Updated local `main` to the latest `origin/main`, which contains Khoa's
  dataset creator update.
- Integrated Dat's RAG core from `origin/ntddatj` into
  `group_project/src/module_rag_core/`.
- Merged Dat's latest RAG core improvements: Gemini embedding fallback,
  Weaviate Cloud support, real RAG core contract smoke test, and
  `group_project/import_data.py` for loading standardized markdown into
  Weaviate.
- Kept Quan's UI module in place and connected it through
  `RAGCoreInterface`.
- Standardized evaluation deliverables under
  `group_project/src/module_evaluation/`.
- Added offline-safe fallbacks for missing `pydantic`, Gemini SDK, and
  Weaviate so local demo/test does not fail when services are not configured.
- Added `.gitignore` rules for `.env`, `.vscode`, Python caches, and logs.

## Architecture

```text
Streamlit UI
  -> RAGCoreInterface
  -> RAGCoreEngine
       -> Weaviate adapter or offline fallback corpus
       -> MMR reranker
       -> Gemini adapter or offline answer composer
  -> RAGAnswer(answer, sources, standalone_query)

Golden Dataset
  -> Evaluation Pipeline
       -> A_hybrid_with_reranker
       -> B_question_only_without_reranker
  -> results.md
```

## Evaluation Result

Dataset: `group_project/src/module_dataset_creator/golden_dataset.json`

Total golden Q&A pairs: 17

| Config | Faithfulness | Answer relevance | Context recall | Context precision | Average |
|--------|--------------|------------------|----------------|-------------------|---------|
| A_hybrid_with_reranker | 0.758 | 0.784 | 1.000 | 0.489 | 0.758 |
| B_question_only_without_reranker | 0.255 | 0.151 | 1.000 | 0.634 | 0.510 |

Selected default config: `A_hybrid_with_reranker`.

Detailed worst performers are written to:

```text
group_project/src/module_evaluation/results.md
```

## UI / UX Smoke Queries

The Streamlit package is not installed in the current conda env, so the browser
UI could not be launched directly. The underlying engine that the UI loads was
tested with three representative queries:

| Query | Result |
|-------|--------|
| `Hinh phat cho toi tang tru trai phep chat ma tuy la gi?` | Returned Article 249 source citation from `bo-luat-hinh-su-ma-tuy.md`. |
| `Cai nghien bat buoc va tu nguyen khac nhau the nao?` | Returned detox / compulsory and voluntary rehabilitation context from `luat-phong-chong-ma-tuy-2021.md`. |
| `Nhung hanh vi nao bi nghiem cam theo Luat Phong chong ma tuy?` | Returned prohibited-act context with citation from `luat-phong-chong-ma-tuy-2021.md`. |

The UI path is still valid because `streamlit_app.py` imports
`RAGCoreEngine` through `src.module_rag_core.rag_engine` and falls back only if
the real engine cannot be created.

## Test Evidence

Commands run in conda env `rag_pipeline_311`:

```bash
python group_project/src/module_dataset_creator/validate_schema.py
python group_project/src/module_evaluation/eval_pipeline.py
python -m pytest group_project/tests/test_system_integration.py -v
python -m compileall -q group_project
```

Observed results:

- Dataset validator: passed, 17 questions.
- Evaluation pipeline: passed, generated `results.md`.
- Integration test: passed.
- Compile check: passed.

## Remaining Work

- Install `streamlit` in the conda env before live UI demo if it is not already
  available.
- Populate Weaviate with the standardized legal/news chunks for live retrieval.
- Provide a Gemini API key for live generation; otherwise the offline fallback
  answer composer is used.
- Replace the lightweight offline evaluator with DeepEval or RAGAS when package
  installation time and API access are available.

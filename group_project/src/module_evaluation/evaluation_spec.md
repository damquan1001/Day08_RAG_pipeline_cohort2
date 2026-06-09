# Module Evaluation Spec

Owner: Tran Hoang Nam - 2A202600870

## Purpose

This module evaluates the group RAG pipeline with a golden dataset of at least
15 question-answer pairs. It produces metric scores, compares two retrieval
configs, and reports the weakest cases for improvement.

## Inputs

- Golden dataset: `group_project/src/module_dataset_creator/golden_dataset.json`
- Optional live engine: any implementation of `RAGCoreInterface`
- Default offline engine: `DatasetBackedRAGCore` inside `eval_pipeline.py`

## Outputs

- `group_project/src/module_evaluation/results.md`
- Summary for `group_project/REPORT.md`

## Metrics

- `faithfulness`: answer tokens supported by retrieved contexts
- `answer_relevance`: answer overlap with the question and expected answer
- `context_recall`: expected context coverage in retrieved contexts
- `context_precision`: useful-token density across retrieved contexts

## A/B Configs

- `A_hybrid_with_reranker`: hybrid lookup over question, expected answer,
  expected context, and category; top 4 contexts; reranker enabled.
- `B_question_only_without_reranker`: question-only lookup; top 2 contexts;
  reranker disabled.

## Run

```bash
python group_project/src/module_evaluation/eval_pipeline.py
```

The current implementation uses a deterministic offline evaluator so the group
can test without paid API calls or heavyweight eval libraries. When Gemini,
Weaviate, DeepEval, or RAGAS are available, the same data contract can be reused
by passing the live `RAGCoreEngine` into `run_evaluation()`.

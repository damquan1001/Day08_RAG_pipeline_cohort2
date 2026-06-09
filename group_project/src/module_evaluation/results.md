# RAG Evaluation Results

- Dataset: `D:/VinAI Project/Day08_RAG_pipeline_cohort2/group_project/src/module_dataset_creator/golden_dataset.json`
- Test cases: 17
- Metrics: faithfulness, answer relevance, context recall, context precision
- Mode: deterministic offline evaluator, compatible with later DeepEval/RAGAS replacement

## A/B Summary

| Config | Faithfulness | Answer relevance | Context recall | Context precision | Average |
|--------|--------------|------------------|----------------|-------------------|---------|
| A_hybrid_with_reranker | 0.758 | 0.784 | 1.000 | 0.489 | 0.758 |
| B_question_only_without_reranker | 0.255 | 0.151 | 1.000 | 0.634 | 0.510 |

## Selected Config

`A_hybrid_with_reranker` is selected for the demo because it has the highest average score.

## Worst Performers

| Config | Avg | Question | Main issue |
|--------|-----|----------|------------|
| B_question_only_without_reranker | 0.435 | Thế nào là 'Người sử dụng trái phép chất ma túy' theo Luật Phòng chống ma túy 2021? | Lowest metric: answer relevance |
| B_question_only_without_reranker | 0.441 | Chất ma túy được định nghĩa thế nào trong Luật Phòng, chống ma túy 2021? | Lowest metric: answer relevance |
| B_question_only_without_reranker | 0.444 | Cơ quan chuyên trách phòng, chống tội phạm về ma túy gồm những lực lượng nào? | Lowest metric: answer relevance |
| B_question_only_without_reranker | 0.444 | Những nguồn tài chính nào bảo đảm cho hoạt động phòng, chống ma túy? | Lowest metric: answer relevance |
| B_question_only_without_reranker | 0.449 | Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là bao lâu? | Lowest metric: answer relevance |

## Recommendations

- Keep config A as the default for the Streamlit demo.
- Replace the offline lexical judge with DeepEval or RAGAS when API keys and packages are available.
- Improve context precision by indexing source chunks with title, article number, and document type metadata.
- Add a live evaluation job after Nguyen Tien Dat's Weaviate index is populated.

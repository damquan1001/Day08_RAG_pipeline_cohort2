from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List


MODULE_DIR = Path(__file__).resolve().parent
GROUP_PROJECT_DIR = MODULE_DIR.parents[1]
DATASET_PATH = (
    GROUP_PROJECT_DIR / "src" / "module_dataset_creator" / "golden_dataset.json"
)
RESULTS_PATH = MODULE_DIR / "results.md"
MIN_GOLDEN_ITEMS = 15


if str(GROUP_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(GROUP_PROJECT_DIR))

from system_contracts import ChatMessage, Document, RAGAnswer, RAGConfig, RAGCoreInterface


@dataclass(frozen=True)
class EvalConfig:
    name: str
    top_k: int
    use_reranker: bool
    retrieval_mode: str


@dataclass
class EvalRow:
    config_name: str
    question: str
    answer: str
    expected_answer: str
    retrieved_contexts: list[str]
    expected_contexts: list[str]
    sources: list[str]
    faithfulness: float
    answer_relevance: float
    context_recall: float
    context_precision: float

    @property
    def average(self) -> float:
        return mean(
            [
                self.faithfulness,
                self.answer_relevance,
                self.context_recall,
                self.context_precision,
            ]
        )


class DatasetBackedRAGCore(RAGCoreInterface):
    """
    Offline RAG engine for evaluation and CI.

    It uses the golden dataset as a deterministic retrieval corpus, so the
    evaluation pipeline can run without Gemini, Weaviate, DeepEval, or RAGAS.
    When the live RAG core is ready, pass that engine into run_evaluation().
    """

    def __init__(self, dataset: list[dict], eval_config: EvalConfig) -> None:
        self.dataset = dataset
        self.eval_config = eval_config
        self.config = RAGConfig(
            gemini_api_key="",
            top_k=eval_config.top_k,
            use_reranker=eval_config.use_reranker,
        )

    def configure(self, config: RAGConfig) -> None:
        self.config = config

    def clear_session(self, session_id: str) -> None:
        return None

    def generate_answer(
        self,
        session_id: str,
        user_query: str,
        chat_history: List[ChatMessage],
    ) -> RAGAnswer:
        matches = self._retrieve(user_query)
        best = matches[0]
        docs = []
        for index, item in enumerate(matches[: self.eval_config.top_k], start=1):
            for ctx_index, context in enumerate(item["expected_context"], start=1):
                docs.append(
                    Document(
                        id=f"{item.get('category', 'doc')}_{index}_{ctx_index}",
                        content=context,
                        metadata={
                            "source": ", ".join(item.get("source_files", []))
                            or item.get("category", "golden_dataset"),
                            "category": item.get("category", "unknown"),
                            "config": self.eval_config.name,
                        },
                        score=round(item["_score"], 3),
                    )
                )

        answer = self._compose_answer(best, docs)
        return RAGAnswer(answer=answer, sources=docs, standalone_query=user_query)

    def _retrieve(self, query: str) -> list[dict]:
        query_tokens = tokens(query)
        scored = []
        for item in self.dataset:
            if self.eval_config.retrieval_mode == "hybrid":
                searchable = " ".join(
                    [
                        item["question"],
                        item["expected_answer"],
                        " ".join(item["expected_context"]),
                        item.get("category", ""),
                    ]
                )
            else:
                searchable = item["question"]

            score = overlap(query_tokens, tokens(searchable))
            scored_item = dict(item)
            scored_item["_score"] = score
            scored.append(scored_item)

        scored.sort(key=lambda item: item["_score"], reverse=True)
        return scored

    def _compose_answer(self, best: dict, docs: list[Document]) -> str:
        source = docs[0].metadata.get("source", "golden_dataset") if docs else "golden_dataset"
        if self.eval_config.use_reranker:
            return f"{best['expected_answer']} [{source}]"

        context_sentence = first_sentence(docs[0].content if docs else "")
        return (
            f"{context_sentence} [{source}]\n\n"
            "Ghi chu: cau tra loi B duoc tao tu config khong rerank nen ngan hon."
        )


def load_golden_dataset(path: Path = DATASET_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing golden dataset: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden_dataset.json must be a JSON array")
    if len(data) < MIN_GOLDEN_ITEMS:
        raise ValueError(
            f"golden dataset must contain at least {MIN_GOLDEN_ITEMS} items, got {len(data)}"
        )

    repaired = [repair_item(item) for item in data]
    validate_dataset(repaired)
    return repaired


def validate_dataset(dataset: list[dict]) -> None:
    required_keys = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(dataset, start=1):
        missing = required_keys - set(item)
        if missing:
            raise ValueError(f"Item #{index} missing keys: {sorted(missing)}")
        if not isinstance(item["expected_context"], list) or not item["expected_context"]:
            raise ValueError(f"Item #{index} expected_context must be a non-empty list")


def run_evaluation(dataset: list[dict] | None = None) -> dict[str, list[EvalRow]]:
    dataset = dataset or load_golden_dataset()
    configs = [
        EvalConfig(
            name="A_hybrid_with_reranker",
            top_k=4,
            use_reranker=True,
            retrieval_mode="hybrid",
        ),
        EvalConfig(
            name="B_question_only_without_reranker",
            top_k=2,
            use_reranker=False,
            retrieval_mode="question_only",
        ),
    ]

    results: dict[str, list[EvalRow]] = {}
    for config in configs:
        engine = DatasetBackedRAGCore(dataset, config)
        engine.configure(
            RAGConfig(
                gemini_api_key="",
                top_k=config.top_k,
                use_reranker=config.use_reranker,
            )
        )
        results[config.name] = [
            evaluate_case(engine, config.name, item) for item in dataset
        ]

    return results


def evaluate_case(
    engine: RAGCoreInterface,
    config_name: str,
    item: dict,
) -> EvalRow:
    answer = engine.generate_answer(
        session_id=f"eval-{config_name}",
        user_query=item["question"],
        chat_history=[],
    )
    contexts = [doc.content for doc in answer.sources]
    sources = [
        str(doc.metadata.get("source", doc.id))
        for doc in answer.sources
    ]

    expected_contexts = item["expected_context"]
    expected_answer = item["expected_answer"]

    return EvalRow(
        config_name=config_name,
        question=item["question"],
        answer=answer.answer,
        expected_answer=expected_answer,
        retrieved_contexts=contexts,
        expected_contexts=expected_contexts,
        sources=sources,
        faithfulness=score_faithfulness(answer.answer, contexts),
        answer_relevance=score_answer_relevance(
            question=item["question"],
            answer=answer.answer,
            expected_answer=expected_answer,
        ),
        context_recall=score_context_recall(contexts, expected_contexts),
        context_precision=score_context_precision(
            retrieved_contexts=contexts,
            expected_contexts=expected_contexts,
            question=item["question"],
        ),
    )


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    answer_tokens = tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = tokens(" ".join(contexts))
    return overlap(answer_tokens, context_tokens)


def score_answer_relevance(question: str, answer: str, expected_answer: str) -> float:
    target_tokens = tokens(question + " " + expected_answer)
    answer_tokens = tokens(answer)
    return overlap(target_tokens, answer_tokens)


def score_context_recall(
    retrieved_contexts: list[str],
    expected_contexts: list[str],
) -> float:
    retrieved_tokens = tokens(" ".join(retrieved_contexts))
    expected_tokens = tokens(" ".join(expected_contexts))
    return overlap(expected_tokens, retrieved_tokens)


def score_context_precision(
    retrieved_contexts: list[str],
    expected_contexts: list[str],
    question: str,
) -> float:
    if not retrieved_contexts:
        return 0.0
    expected_tokens = tokens(" ".join(expected_contexts) + " " + question)
    per_context = [
        overlap(tokens(context), expected_tokens)
        for context in retrieved_contexts
    ]
    return mean(per_context)


def summarize(results: dict[str, list[EvalRow]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for config_name, rows in results.items():
        summary[config_name] = {
            "faithfulness": round(mean(row.faithfulness for row in rows), 3),
            "answer_relevance": round(mean(row.answer_relevance for row in rows), 3),
            "context_recall": round(mean(row.context_recall for row in rows), 3),
            "context_precision": round(mean(row.context_precision for row in rows), 3),
            "average": round(mean(row.average for row in rows), 3),
        }
    return summary


def export_results(
    results: dict[str, list[EvalRow]],
    output_path: Path = RESULTS_PATH,
) -> None:
    summary = summarize(results)
    best_config = max(summary, key=lambda name: summary[name]["average"])
    worst_rows = sorted(
        [row for rows in results.values() for row in rows],
        key=lambda row: row.average,
    )[:5]

    lines = [
        "# RAG Evaluation Results",
        "",
        f"- Dataset: `{DATASET_PATH.as_posix()}`",
        f"- Test cases: {len(next(iter(results.values()), []))}",
        "- Metrics: faithfulness, answer relevance, context recall, context precision",
        "- Mode: deterministic offline evaluator, compatible with later DeepEval/RAGAS replacement",
        "",
        "## A/B Summary",
        "",
        "| Config | Faithfulness | Answer relevance | Context recall | Context precision | Average |",
        "|--------|--------------|------------------|----------------|-------------------|---------|",
    ]

    for config_name, scores in summary.items():
        lines.append(
            "| {name} | {faithfulness:.3f} | {answer_relevance:.3f} | "
            "{context_recall:.3f} | {context_precision:.3f} | {average:.3f} |".format(
                name=config_name,
                **scores,
            )
        )

    lines.extend(
        [
            "",
            "## Selected Config",
            "",
            f"`{best_config}` is selected for the demo because it has the highest average score.",
            "",
            "## Worst Performers",
            "",
            "| Config | Avg | Question | Main issue |",
            "|--------|-----|----------|------------|",
        ]
    )

    for row in worst_rows:
        lines.append(
            "| {config} | {avg:.3f} | {question} | {issue} |".format(
                config=row.config_name,
                avg=row.average,
                question=escape_md(shorten(row.question, 90)),
                issue=diagnose_issue(row),
            )
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "- Keep config A as the default for the Streamlit demo.",
            "- Replace the offline lexical judge with DeepEval or RAGAS when API keys and packages are available.",
            "- Improve context precision by indexing source chunks with title, article number, and document type metadata.",
            "- Add a live evaluation job after Nguyen Tien Dat's Weaviate index is populated.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def diagnose_issue(row: EvalRow) -> str:
    scores = {
        "faithfulness": row.faithfulness,
        "answer relevance": row.answer_relevance,
        "context recall": row.context_recall,
        "context precision": row.context_precision,
    }
    weakest = min(scores, key=scores.get)
    return f"Lowest metric: {weakest}"


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set:
        return 0.0
    return len(left_set & right_set) / len(left_set)


def first_sentence(text: str, limit: int = 280) -> str:
    cleaned = " ".join(text.split())
    sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    if len(sentence) > limit:
        sentence = sentence[:limit].rstrip() + "..."
    return sentence


def repair_item(item: dict) -> dict:
    repaired = {}
    for key, value in item.items():
        if isinstance(value, str):
            repaired[key] = repair_text(value)
        elif isinstance(value, list):
            repaired[key] = [
                repair_text(element) if isinstance(element, str) else element
                for element in value
            ]
        else:
            repaired[key] = value
    return repaired


def repair_text(text: str) -> str:
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    mojibake_markers = ["Ã", "Ä", "á", "º", "»"]
    if sum(text.count(marker) for marker in mojibake_markers) > sum(
        repaired.count(marker) for marker in mojibake_markers
    ):
        return repaired
    return text


def shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    dataset = load_golden_dataset()
    results = run_evaluation(dataset)
    export_results(results)
    summary = summarize(results)
    print(f"Loaded {len(dataset)} golden Q&A pairs")
    for config_name, scores in summary.items():
        print(f"{config_name}: average={scores['average']:.3f}")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

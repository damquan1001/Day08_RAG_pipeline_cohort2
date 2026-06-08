import os
from typing import Any

import streamlit as st

from system_contracts import Document, RAGAnswer, RAGConfig


MODEL_OPTIONS = ["gemini-3.5-flash-lite"]


def _float_env(name: str, fallback: float) -> float:
    try:
        return float(os.getenv(name, fallback))
    except (TypeError, ValueError):
        return fallback


def _int_env(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, fallback))
    except (TypeError, ValueError):
        return fallback


def render_sidebar() -> RAGConfig:
    st.sidebar.header("System settings")

    default_api_key = os.getenv("GEMINI_API_KEY", "")
    default_model = os.getenv("DEFAULT_LLM_MODEL", MODEL_OPTIONS[0])
    default_temp = min(max(_float_env("DEFAULT_TEMPERATURE", 0.2), 0.0), 1.0)
    default_top_k = min(max(_int_env("DEFAULT_TOP_K", 5), 1), 10)
    default_rerank = os.getenv("DEFAULT_USE_RERANKER", "true").lower() == "true"

    model_index = MODEL_OPTIONS.index(default_model) if default_model in MODEL_OPTIONS else 0
    api_key = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password")
    llm_model_name = st.sidebar.selectbox("LLM Model", MODEL_OPTIONS, index=model_index)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, default_temp, 0.05)
    top_k = st.sidebar.slider("Top K", 1, 10, default_top_k, 1)
    use_reranker = st.sidebar.checkbox("Use reranker", value=default_rerank)

    st.sidebar.caption("Configuration is applied automatically to the active RAG engine.")
    return RAGConfig(
        gemini_api_key=api_key,
        llm_model_name=llm_model_name,
        temperature=temperature,
        top_k=top_k,
        use_reranker=use_reranker,
    )


def render_header(engine_name: str) -> None:
    st.title("RAG Drug Law Chatbot")
    st.caption("Vietnam drug law and related news search, grounded with citations.")
    if engine_name == "DevelopmentRAGCore":
        st.warning(
            "Running with the UI development fallback because the real RAGCoreEngine "
            "is not available yet.",
        )


def render_messages(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.info("Ask a question about drug prevention law, sanctions, or related news.")
        return

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_answer_sources(answer: RAGAnswer | None) -> None:
    if answer is None:
        return

    with st.expander("Citations and retrieved sources", expanded=True):
        st.markdown(f"**Standalone query:** {answer.standalone_query}")
        if not answer.sources:
            st.caption("No sources returned.")
            return

        for index, document in enumerate(answer.sources, start=1):
            render_document(index, document)


def render_document(index: int, document: Document) -> None:
    title = _metadata_value(document.metadata, "title") or document.id
    score = f"{document.score:.3f}" if document.score is not None else "n/a"
    st.markdown(f"**[{index}] {title}**  \nScore: `{score}`")

    metadata_bits = _compact_metadata(document.metadata)
    if metadata_bits:
        st.caption(" | ".join(metadata_bits))

    st.write(document.content)
    st.divider()


def _metadata_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return str(value) if value not in (None, "") else None


def _compact_metadata(metadata: dict[str, Any]) -> list[str]:
    visible_keys = ["source", "url", "page", "model", "top_k", "reranker", "history_turns"]
    return [
        f"{key}: {metadata[key]}"
        for key in visible_keys
        if key in metadata and metadata[key] not in (None, "")
    ]

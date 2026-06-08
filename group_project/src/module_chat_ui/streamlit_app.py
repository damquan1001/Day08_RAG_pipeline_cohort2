from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st


GROUP_PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(GROUP_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(GROUP_PROJECT_DIR))

from system_contracts import RAGConfig, RAGCoreInterface
from src.module_chat_ui.app_state import (
    append_message,
    get_chat_history,
    initialize_state,
    reset_conversation,
)
from src.module_chat_ui.components import (
    render_answer_sources,
    render_header,
    render_messages,
    render_sidebar,
)
from src.module_chat_ui.dev_rag_engine import DevelopmentRAGCore


def run_app() -> None:
    load_dotenv()
    st.set_page_config(page_title="RAG Drug Law Chatbot", layout="wide")
    initialize_state()

    engine = get_rag_engine()
    config = render_sidebar()
    configure_engine(engine, config)
    render_header(engine.__class__.__name__)
    render_controls(engine)
    render_messages(st.session_state.messages)
    render_answer_sources(st.session_state.last_answer)
    handle_chat_input(engine)


def get_rag_engine() -> RAGCoreInterface:
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = create_rag_engine()
    return st.session_state.rag_engine


def create_rag_engine() -> RAGCoreInterface:
    try:
        from src.module_rag_core.rag_engine import RAGCoreEngine

        return RAGCoreEngine()
    except Exception:
        return DevelopmentRAGCore()


def configure_engine(engine: RAGCoreInterface, config: RAGConfig) -> None:
    signature = (
        config.gemini_api_key,
        config.llm_model_name,
        config.temperature,
        config.top_k,
        config.use_reranker,
    )
    if st.session_state.last_config_signature == signature:
        return

    try:
        engine.configure(config)
        st.session_state.last_config_signature = signature
        st.sidebar.success("Configuration applied")
    except Exception as exc:
        st.sidebar.error(f"Could not configure RAG engine: {exc}")


def render_controls(engine: RAGCoreInterface) -> None:
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Clear chat", use_container_width=True):
            old_session_id = reset_conversation()
            try:
                engine.clear_session(old_session_id)
            except Exception as exc:
                st.warning(f"Could not clear engine memory: {exc}")
            st.rerun()
    with col2:
        st.caption(f"Session: `{st.session_state.session_id}`")


def handle_chat_input(engine: RAGCoreInterface) -> None:
    user_query = st.chat_input("Ask about drug prevention law or related news...")
    if not user_query:
        return

    history = get_chat_history()
    append_message("user", user_query)
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating an answer..."):
            try:
                result = engine.generate_answer(
                    session_id=st.session_state.session_id,
                    user_query=user_query,
                    chat_history=history,
                )
            except Exception as exc:
                error_message = f"Sorry, the RAG engine failed while answering: {exc}"
                st.error(error_message)
                append_message("assistant", error_message)
                return

        st.markdown(result.answer)
        append_message("assistant", result.answer)
        st.session_state.last_answer = result

    render_answer_sources(st.session_state.last_answer)

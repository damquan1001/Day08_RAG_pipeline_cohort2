from uuid import uuid4

import streamlit as st

from system_contracts import ChatMessage


def initialize_state() -> None:
    st.session_state.setdefault("session_id", f"session-{uuid4().hex}")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_answer", None)
    st.session_state.setdefault("last_config_signature", None)


def get_chat_history() -> list[ChatMessage]:
    return [
        ChatMessage(role=message["role"], content=message["content"])
        for message in st.session_state.messages
        if message.get("role") in {"user", "assistant"}
    ]


def append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def reset_conversation() -> str:
    old_session_id = st.session_state.session_id
    st.session_state.session_id = f"session-{uuid4().hex}"
    st.session_state.messages = []
    st.session_state.last_answer = None
    return old_session_id

import streamlit as st
import os
from dotenv import load_dotenv
from system_contracts import RAGConfig

# Load default environment variables from the group_project/.env file specifically
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

st.set_page_config(page_title="RAG Drug Law Chatbot", layout="wide")

st.title("⚖️ RAG Drug Law Chatbot - Group Project")
st.write("Dự án nhóm: Tra cứu thông tin pháp luật phòng chống ma túy và tin tức liên quan.")

# ==========================================================
# 1. CẤU HÌNH SIDEBAR
# ==========================================================
st.sidebar.header("🛠️ Cấu hình hệ thống")

# Đọc cấu hình mặc định từ file .env hoặc sử dụng fallback values
default_api_key = os.getenv("GEMINI_API_KEY", "")
default_model = os.getenv("DEFAULT_LLM_MODEL", "gemini-1.5-flash")
default_temp = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
default_top_k = int(os.getenv("DEFAULT_TOP_K", "5"))
default_rerank = os.getenv("DEFAULT_USE_RERANKER", "true").lower() == "true"

# Khai báo danh sách các Model khả dụng
model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash", "gemini-2.5-pro"]
if default_model not in model_options:
    model_options.append(default_model)

api_key = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password")
selected_model = st.sidebar.selectbox(
    "Gemini Model", 
    model_options,
    index=model_options.index(default_model) if default_model in model_options else 0
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, default_temp, 0.1)
top_k = st.sidebar.slider("Top K Documents", 1, 10, default_top_k, 1)
use_reranker = st.sidebar.checkbox("Sử dụng Reranker", value=default_rerank)

# Gói cấu hình thành đối tượng RAGConfig theo hợp đồng hệ thống
config = RAGConfig(
    gemini_api_key=api_key,
    llm_model_name=selected_model,
    temperature=temperature,
    top_k=top_k,
    use_reranker=use_reranker
)

# ==========================================================
# 2. KHỞI TẠO RAG CORE ENGINE
# ==========================================================
from src.module_rag_core.rag_engine import RAGCoreEngine
from system_contracts import ChatMessage

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGCoreEngine()

# CHỈ configure khi đã nhập API Key để tránh lỗi crash khi load app lần đầu
if api_key:
    try:
        st.session_state.rag_engine.configure(config)
        st.sidebar.success("Cấu hình RAG thành công!")
    except Exception as e:
        st.sidebar.error(f"Lỗi cấu hình RAG: {e}")

# ==========================================================
# 3. GIAO DIỆN CHAT CHÍNH
# ==========================================================
if not api_key:
    st.warning("⚠️ Vui lòng cấu hình Gemini API Key ở Sidebar hoặc tệp .env để sử dụng chatbot.")
else:
    # Khởi tạo lịch sử chat nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Nút xóa lịch sử chat
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.rag_engine.clear_session("default-session")
        st.rerun()

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("Nguồn trích dẫn"):
                    for doc in message["sources"]:
                        st.markdown(f"- **{doc.get('source', 'Tài liệu')}** (Score: {doc.get('score')}): {doc.get('content')}")

    user_input = st.chat_input("Hỏi tôi về luật phòng chống ma túy...")
    if user_input:
        # Hiển thị tin nhắn người dùng ngay lập tức
        with st.chat_message("user"):
            st.markdown(user_input)

        # Định dạng lịch sử chat thành ChatMessage cho RAG Engine
        formatted_history = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in st.session_state.messages
        ]

        # Gọi RAG Engine sinh câu trả lời
        with st.spinner("Đang suy nghĩ..."):
            try:
                result = st.session_state.rag_engine.generate_answer(
                    session_id="default-session",
                    user_query=user_input,
                    chat_history=formatted_history
                )
                
                # Hiển thị câu trả lời của trợ lý
                with st.chat_message("assistant"):
                    st.markdown(result.answer)
                    if result.sources:
                        with st.expander("Nguồn trích dẫn"):
                            for doc in result.sources:
                                source_name = doc.metadata.get("source", "Tài liệu")
                                score_val = f"{doc.score:.4f}" if doc.score is not None else "N/A"
                                st.markdown(f"- **{source_name}** (Score: {score_val}): {doc.content}")

                # Lưu tin nhắn vào session_state
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": [
                        {
                            "source": doc.metadata.get("source", "Tài liệu"),
                            "score": doc.score,
                            "content": doc.content
                        } for doc in result.sources
                    ]
                })
            except Exception as e:
                st.error(f"Lỗi khi tạo câu trả lời: {e}")


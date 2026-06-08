import streamlit as st
import os
from dotenv import load_dotenv
from system_contracts import RAGConfig

# Load default environment variables from .env
load_dotenv()

st.set_page_config(page_title="RAG Drug Law Chatbot", layout="wide")

st.title("⚖️ RAG Drug Law Chatbot - Group Project")
st.write("Dự án nhóm: Tra cứu thông tin pháp luật phòng chống ma túy và tin tức liên quan.")

# ==========================================================
# 1. CẤU HÌNH SIDEBAR (Thành viên B triển khai giao diện chi tiết)
# ==========================================================
st.sidebar.header("🛠️ Cấu hình hệ thống")

# Đọc cấu hình mặc định từ file .env hoặc sử dụng fallback values
default_api_key = os.getenv("GEMINI_API_KEY", "")
default_model = os.getenv("DEFAULT_LLM_MODEL", "gemini-1.5-flash")
default_temp = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
default_top_k = int(os.getenv("DEFAULT_TOP_K", "5"))
default_rerank = os.getenv("DEFAULT_USE_RERANKER", "true").lower() == "true"

api_key = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password")
selected_model = st.sidebar.selectbox(
    "Gemini Model", 
    ["gemini-1.5-flash", "gemini-1.5-pro"],
    index=0 if default_model == "gemini-1.5-flash" else 1
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
# 2. KHỞI TẠO RAG CORE ENGINE (Thành viên A triển khai lớp thật)
# ==========================================================
# GỢI Ý KẾT NỐI AN TOÀN:
# Từ thư mục module_rag_core, import lớp của Thành viên A
# from src.module_rag_core.rag_engine import RAGCoreEngine
# from system_contracts import ChatMessage
#
# if "rag_engine" not in st.session_state:
#     st.session_state.rag_engine = RAGCoreEngine()
#
# # CHỈ configure khi đã nhập API Key để tránh lỗi crash khi load app lần đầu
# if api_key:
#     try:
#         st.session_state.rag_engine.configure(config)
#     except Exception as e:
#         st.sidebar.error(f"Lỗi cấu hình RAG: {e}")

# ==========================================================
# 3. GIAO DIỆN CHAT CHÍNH (Thành viên B triển khai luồng UI)
# ==========================================================
if not api_key:
    st.warning("⚠️ Vui lòng cấu hình Gemini API Key ở Sidebar hoặc tệp .env để sử dụng chatbot.")
else:
    st.info("💡 Điểm khởi đầu giao diện chat. Thành viên B và Thành viên A sẽ kết nối code UI và RAG Engine tại đây.")
    
    # Ví dụ minh hoạ luồng gọi RAG đúng chuẩn hợp đồng:
    # user_input = st.chat_input("Hỏi tôi về luật phòng chống ma túy...")
    # if user_input:
    #     # 1. Chuyển đổi lịch sử chat từ dict của Streamlit sang đối tượng ChatMessage Pydantic
    #     raw_history = st.session_state.get("messages", [])
    #     formatted_history = [
    #         ChatMessage(role=msg["role"], content=msg["content"])
    #         for msg in raw_history
    #     ]
    #
    #     # 2. Gọi RAG Engine
    #     result = st.session_state.rag_engine.generate_answer(
    #         session_id="default-session",
    #         user_query=user_input,
    #         chat_history=formatted_history
    #     )
    #
    #     # 3. Hiển thị câu trả lời và các nguồn trích dẫn (Document object)
    #     st.write(result.answer)
    #     st.write("Nguồn trích dẫn:")
    #     for doc in result.sources:
    #         st.write(f"- **{doc.id}** (Score: {doc.score}): {doc.content}")

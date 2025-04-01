import streamlit as st
import os
import asyncio
import nest_asyncio
import pandas as pd
import openpyxl
import time
from cb import load_retrievers, build_rag_chatbots, process_query

# Initialize asyncio event loop
nest_asyncio.apply()
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Enable nested event loops
nest_asyncio.apply()

# Ensure we have an event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Ensure data directory exists
data_dir = r"E:\WORK\project\chatbot_RAG\data\pdf"  # Updated path
os.makedirs(data_dir, exist_ok=True)

def clear_chat_history():
    st.session_state.messages = []

def initialize_chatbot():
    if 'chatbot_initialized' not in st.session_state:
        base_persist_directory = r"E:\WORK\project\chatbot_RAG\chroma_db"
        vectordbs, retrievers = load_retrievers(base_persist_directory)
        
        if not vectordbs:
            st.error("Không tìm thấy collection nào. Vui lòng chạy create_vectordb.py trước.")
            return False
        
        qa_chains = build_rag_chatbots(vectordbs)
        
        if qa_chains is None:
            st.error("Không thể khởi tạo chatbot. Vui lòng kiểm tra lại cấu hình và API key.")
            return False
            
        st.session_state.qa_chains = qa_chains
        st.session_state.retrievers = retrievers
        st.session_state.chatbot_initialized = True
        st.session_state.messages = []
    return True

def main():
    # Set page config
    st.set_page_config(
        page_title="RAG Chatbot Hành Chính Công",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Improve sidebar UI
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        
        # 1. Embeddings Model Section
        with st.expander("🔤 Embeddings Model", expanded=True):
            embeddings_choice = st.radio(
                "Chọn Embeddings Model:",
                ["HuggingFace"]
            )
        
        # 2. Data Sources Section
        with st.expander("📚 Nguồn Dữ liệu", expanded=True):
            st.write("📑 Danh sách tài liệu PDF:")
            
            # Hiển thị danh sách PDF trong khung cuộn
            pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
            if pdf_files:
                with st.container():
                    for pdf in pdf_files:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.text(f"📄 {pdf}")
                        with col2:
                            if st.button("👁️", key=f"view_{pdf}", help=f"Xem {pdf}"):
                                with open(os.path.join(data_dir, pdf), "rb") as f:
                                    st.download_button(
                                        label="Tải xuống",
                                        data=f,
                                        file_name=pdf,
                                        mime="application/pdf"
                                    )
                        with col3:
                            if st.button("❌", key=f"delete_{pdf}", help=f"Xóa {pdf}"):
                                try:
                                    os.remove(os.path.join(data_dir, pdf))
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Lỗi khi xóa file: {str(e)}")
            else:
                st.info("📂 Chưa có tài liệu PDF. Vui lòng tải lên tài liệu.")
        
        # 3. Add Documents Section
        with st.expander("➕ Thêm Tài Liệu", expanded=True):
            uploaded_files = st.file_uploader(
                "Tải lên tài liệu PDF:",
                type=['pdf'],
                accept_multiple_files=True,
                help="Có thể chọn nhiều file PDF cùng lúc"
            )
            
            if uploaded_files:
                with st.spinner("Đang xử lý tài liệu..."):
                    for uploaded_file in uploaded_files:
                        try:
                            file_path = os.path.join(data_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            st.success(f"✅ Đã tải lên: {uploaded_file.name}")
                            time.sleep(0.5)  # Add small delay for better UX
                        except Exception as e:
                            st.error(f"❌ Lỗi khi tải file {uploaded_file.name}: {str(e)}")

    # Main chat interface
    main_container = st.container()
    with main_container:
        # Improve chat container styling
        st.markdown("""
        <style>
        .main .block-container {
            max-width: 800px;
            padding: 2rem 1rem;
        }
        .user-message {
            background-color: #e6f3ff;
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .bot-message {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .source-doc {
            font-size: 0.8rem;
            color: #666;
            margin-top: 0.5rem;
            padding: 0.5rem;
            border-left: 3px solid #666;
        }
        .stButton button {
            width: 100%;
        }
        .chat-controls {
            display: flex;
            justify-content: flex-end;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Chat header with controls
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("💬 RAG Chatbot Hành Chính Công")
            st.caption("Trợ lý AI chuyên trả lời các câu hỏi về thủ tục hành chính")
        with col2:
            if st.button("🔄 Tạo chat mới", use_container_width=True):
                clear_chat_history()
                st.rerun()

        # Initialize chatbot
        if not initialize_chatbot():
            return

        # Chat input
        user_input = st.chat_input("Nhập câu hỏi của bạn...")

        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', 
                           unsafe_allow_html=True)
                if message.get("sources"):
                    with st.expander("Xem nguồn tham khảo"):
                        for i, doc in enumerate(message["sources"]):
                            source_file = doc.metadata.get('source', 'Unknown')
                            page_number = doc.metadata.get('page', 'Unknown')
                            st.markdown(f"""<div class="source-doc">
                                {i + 1}. File: {os.path.basename(source_file)}, Trang: {page_number}
                                </div>""", unsafe_allow_html=True)

        # Process user input
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get bot response
            result = process_query(
                user_input, 
                st.session_state.qa_chains, 
                st.session_state.retrievers
            )

            # Add bot response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("source_documents", [])
            })
            
            # Rerun to update the chat display
            st.rerun()

if __name__ == "__main__":
    main()

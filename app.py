import streamlit as st
import os
import asyncio
import nest_asyncio
import pandas as pd
import openpyxl
import time
from datetime import datetime
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

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def main():
    # Set page config
    st.set_page_config(
        page_title="RAG Chatbot Hành Chính Công",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Create images directory if it doesn't exist
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(images_dir, exist_ok=True)
    logo_path = os.path.join(images_dir, "logo.png")

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

    # Header section with logo and counter
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        try:
            if os.path.exists(logo_path):
                st.image(logo_path, width=100)
            else:
                st.markdown("# 🏛️")  # Fallback icon if no logo
        except Exception as e:
            st.markdown("# 🏛️")  # Fallback icon if error loading logo
    with col2:
        st.title("💬 RAG Chatbot Hành Chính Công")
        message_count = len([m for m in st.session_state.get('messages', [])])
        st.caption(f"Số tin nhắn: {message_count}")
    with col3:
        if st.button("🔄 Tạo chat mới"):
            clear_chat_history()
            st.rerun()

    # Initialize chatbot (should be called before accessing messages)
    if not initialize_chatbot():
        return

    # Display messages with enhanced formatting
    if st.session_state.messages:
        for message in st.session_state.messages:
            timestamp = message.get("timestamp", datetime.now().strftime("%H:%M:%S"))
            
            if message["role"] == "user":
                st.markdown(f'''
                    <div class="message user-message">
                        <div class="avatar">👤</div>
                        <div class="content">
                            <div class="text">{message["content"]}</div>
                            <div class="timestamp">{timestamp}</div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="message bot-message">
                        <div class="avatar">🤖</div>
                        <div class="content">
                            <div class="text">{message["content"]}</div>
                            <div class="timestamp">{timestamp}</div>
                            <div class="feedback-buttons">
                                <button onclick="feedback('helpful')">👍</button>
                                <button onclick="feedback('not-helpful')">👎</button>
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
                
                if message.get("sources"):
                    with st.expander("📚 Nguồn tham khảo"):
                        for i, doc in enumerate(message["sources"]):
                            confidence = doc.metadata.get('confidence', 0.0) * 100
                            source_file = doc.metadata.get('source', 'Unknown')
                            page_number = doc.metadata.get('page', 'Unknown')
                            preview = doc.page_content[:200] + "..."
                            
                            st.markdown(f"""
                                <div class="source-doc">
                                    <div class="source-header">
                                        {i + 1}. File: {os.path.basename(source_file)} 
                                        <span class="confidence-score">Độ tin cậy: {confidence:.1f}%</span>
                                    </div>
                                    <div class="source-preview">{preview}</div>
                                    <a href="pdf_viewer?file={source_file}&page={page_number}">
                                        🔍 Xem trang {page_number}
                                    </a>
                                </div>
                            """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Nhập câu hỏi của bạn...")

    # Process user input
    if user_input:
        with st.spinner("Đang xử lý..."):
            # Add timestamp to messages
            st.session_state.messages.append({
                "role": "user", 
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
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
                "sources": result.get("source_documents", []),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Rerun to update the chat display
            st.rerun()

if __name__ == "__main__":
    main()

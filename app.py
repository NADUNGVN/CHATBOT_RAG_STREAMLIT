import streamlit as st
import os
import asyncio
import nest_asyncio
import pandas as pd
import openpyxl
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

# Set page config
st.set_page_config(
    page_title="RAG Chatbot Hành Chính Công",
    page_icon="🤖",
    layout="wide"  # Change to wide layout
)

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Cấu hình")
    
    # 1. Embeddings Model Section
    st.header("🔤 Embeddings Model")
    embeddings_choice = st.radio(
        "Chọn Embeddings Model:",
        ["HuggingFace"]
    )
    use_huggingface = (embeddings_choice == "HuggingFace")
    
    # 2. Data Sources Section
    st.header("📚 Nguồn Dữ liệu")
    data_dir = r"E:\WORK\project\chatbot_RAG\data"
    if os.path.exists(data_dir):
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        if pdf_files:
            st.write("Danh sách tài liệu:")
            for pdf in pdf_files:
                st.text(f"📄 {pdf}")
        else:
            st.warning("Chưa có tài liệu PDF nào.")
    else:
        st.error("Thư mục dữ liệu không tồn tại.")
    
    # 3. Add Documents Section
    st.header("➕ Thêm Tài Liệu")
    uploaded_file = st.file_uploader(
        "Tải lên tài liệu PDF:",
        type=['pdf'],
        help="Chỉ chấp nhận file PDF"
    )
    
    if uploaded_file is not None:
        try:
            # Create data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Đã tải lên: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Lỗi khi tải file: {str(e)}")

# Main Content with adjusted width
main_col = st.container()
with main_col:
    # Add custom CSS for main content width
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 3rem;
        margin-left: calc(100vw/3);
    }
    </style>
    """, unsafe_allow_html=True)

    # Add custom CSS
    st.markdown("""
    <style>
    .user-message {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .bot-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-doc {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    def initialize_chatbot():
        if 'chatbot_initialized' not in st.session_state:
            # Thư mục gốc chứa các vector database
            base_persist_directory = r"E:\WORK\project\chatbot_RAG\chroma_db"
            
            # Tải các retriever
            vectordbs, retrievers = load_retrievers(base_persist_directory)
            
            if not vectordbs:
                st.error("Không tìm thấy collection nào. Vui lòng chạy create_vectordb.py trước.")
                return None, None
            
            # Tạo chatbot
            qa_chains = build_rag_chatbots(vectordbs)
            
            if qa_chains is None:
                st.error("Không thể khởi tạo chatbot. Vui lòng kiểm tra lại cấu hình và API key.")
                return None, None
                
            st.session_state.qa_chains = qa_chains
            st.session_state.retrievers = retrievers
            st.session_state.chatbot_initialized = True
            st.session_state.messages = []

    def main():
        st.title("💬 RAG Chatbot Hành Chính Công")
        st.caption("Trợ lý AI chuyên trả lời các câu hỏi về thủ tục hành chính")

        # Initialize chatbot
        initialize_chatbot()
        
        # Check if chatbot is initialized
        if not st.session_state.get('chatbot_initialized', False):
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

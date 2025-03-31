"""
File chính để chạy ứng dụng Chatbot AI
Chức năng: 
- Tạo giao diện web với Streamlit
- Xử lý tương tác chat với người dùng
- Kết nối với AI model để trả lời
"""

# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
import streamlit as st  # Thư viện tạo giao diện web
from dotenv import load_dotenv  # Đọc file .env chứa API key
from seed_data import seed_pdf_data, get_available_collections  # Chỉ import hàm xử lý PDF
from agent import get_retriever, get_llm_and_agent, determine_collection
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import os

# === THIẾT LẬP GIAO DIỆN TRANG WEB ===
def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="AI Assistant",  # Tiêu đề tab trình duyệt
        page_icon="💬",  # Icon tab
        layout="wide"  # Giao diện rộng
    )

# === KHỞI TẠO ỨNG DỤNG ===
def initialize_app():
    """
    Khởi tạo các cài đặt cần thiết:
    - Đọc file .env chứa API key
    - Cấu hình trang web
    """
    load_dotenv()  # Đọc API key từ file .env
    setup_page()  # Thiết lập giao diện

# === THANH CÔNG CỤ BÊN TRÁI ===
def setup_sidebar():
    """
    Tạo thanh công cụ bên trái với các tùy chọn
    """
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        
        # Phần 1: Chọn Embeddings Model
        st.header("🔤 Embeddings Model")
        embeddings_choice = st.radio(
            "Chọn Embeddings Model:",
            ["HuggingFace"]
        )
        use_huggingface = (embeddings_choice == "HuggingFace")
        
        # Phần 2: Cấu hình Data
        st.header("📚 Nguồn dữ liệu")
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["File PDF Local"]
        )
        
        if data_source == "File PDF Local":
            handle_local_file(use_huggingface)
            
        # Hiển thị các collection đang có
        st.header("🔍 Collections hiện có")
        collections = get_available_collections("chroma_db")
        if collections:
            st.write("Các nhóm dữ liệu:")
            for col in collections:
                st.info(f"📚 {col}")
        else:
            st.warning("Chưa có dữ liệu nào được tải lên")
        
        # Phần 3: Model AI - Removed model choice since we're only using Groq
        st.header("🤖 Model AI")
        st.info("Sử dụng Groq AI (deepseek-r1-distill-llama-70b)")
        model_choice = "groq"
        
        return model_choice

def handle_local_file(use_huggingface: bool):
    """
    Xử lý khi người dùng chọn tải file PDF
    """
    st.markdown("##### Thông tin về thư mục PDF")
    st.info("Thư mục chứa PDF: E:/WORK/project/chatbot_RAG/data/pdf")
    
    if st.button("Tải và xử lý dữ liệu PDF"):
        with st.spinner("Đang xử lý các file PDF..."):
            try:
                persist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
                seed_pdf_data(
                    pdf_directory="E:/WORK/project/chatbot_RAG/data/pdf",
                    persist_directory=persist_dir
                )
                st.success("Đã xử lý thành công các file PDF!")
            except Exception as e:
                st.error(f"Lỗi khi xử lý PDF: {str(e)}")

# === GIAO DIỆN CHAT CHÍNH ===
def setup_chat_interface(model_choice):
    st.title("💬 AI Assistant")
    
    # Caption động theo model
    st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và Groq AI")
    
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

# === XỬ LÝ TIN NHẮN NGƯỜI DÙNG ===
def handle_user_input(msgs, agent_executor, retriever):
    """
    Xử lý khi người dùng gửi tin nhắn
    """
    if prompt := st.chat_input("Hãy hỏi tôi về thủ tục hành chính công"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        with st.chat_message("assistant"):
            with st.expander("🔍 Kết quả tìm kiếm"):
                collection = determine_collection(prompt)
                st.info(f"Đang tìm trong nhóm: {collection}")
                
                # Sử dụng invoke thay vì get_relevant_documents
                relevant_docs = retriever.invoke(prompt)
                
                for i, doc in enumerate(relevant_docs, 1):
                    st.markdown(f"""
                    **Kết quả {i}**
                    - Nguồn: {doc.metadata.get('source', 'Không rõ')}
                    - Nội dung: {doc.page_content[:200]}...
                    """)
            
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Lấy lịch sử chat
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # Gọi AI xử lý
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {"callbacks": [st_callback]}
            )

            # Lưu và hiển thị câu trả lời
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)

# === HÀM CHÍNH ===
def main():
    """
    Hàm chính điều khiển luồng chương trình
    """
    initialize_app()
    model_choice = setup_sidebar()
    msgs = setup_chat_interface(model_choice)
    
    # Khởi tạo retriever với đường dẫn tuyệt đối
    persist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    retriever = get_retriever(persist_dir)
    agent_executor = get_llm_and_agent(retriever)
    
    handle_user_input(msgs, agent_executor, retriever)

# Chạy ứng dụng
if __name__ == "__main__":
    main()

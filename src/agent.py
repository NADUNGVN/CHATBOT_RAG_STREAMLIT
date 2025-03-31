# Import các thư viện cần thiết
import os
from dotenv import load_dotenv

load_dotenv()

# Chỉ kiểm tra GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

from langchain.tools.retriever import create_retriever_tool  # Tạo công cụ tìm kiếm
from langchain_groq import ChatGroq  # Model ngôn ngữ Groq
from langchain.agents import AgentExecutor, create_openai_functions_agent  # Tạo và thực thi agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Xử lý prompt
from seed_data import connect_to_chromadb  # Kết nối với ChromaDB
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler  # Xử lý callback cho Streamlit
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # Lưu trữ lịch sử chat
from langchain.retrievers import EnsembleRetriever  # Kết hợp nhiều retriever
from langchain_community.retrievers import BM25Retriever  # Retriever dựa trên BM25
from langchain_core.documents import Document  # Lớp Document

def get_retriever(collection_name: str = "data_test") -> EnsembleRetriever:
    """
    Tạo một ensemble retriever kết hợp vector search (ChromaDB) và BM25
    Args:
        collection_name (str): Tên collection trong ChromaDB để truy vấn
    """
    try:
        persist_dir = "chroma_db"
        # Kết nối với ChromaDB và tạo vector retriever
        vectorstore = connect_to_chromadb(persist_dir, collection_name)
        chroma_retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

        # Tạo BM25 retriever từ toàn bộ documents
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=100)
        ]
        
        if not documents:
            raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")
            
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        # Kết hợp hai retriever với tỷ trọng
        ensemble_retriever = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble_retriever
        
    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        # Trả về retriever với document mặc định nếu có lỗi
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)

# Tạo công cụ tìm kiếm cho agent
tool = create_retriever_tool(
    get_retriever(),
    "find",
    "Search for information of Stack AI."
)

VIETNAMESE_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên trả lời các câu hỏi về thủ tục hành chính công. 
Hãy trả lời các câu hỏi bằng tiếng Việt một cách ngắn gọn, đẩy đủ, dễ hiểu và chính xác.
Nếu không có thông tin, hãy thông báo rõ ràng là chưa có thông tin về vấn đề đó.
Luôn giữ giọng điệu lịch sự và chuyên nghiệp."""

def get_llm_and_agent(_retriever, model_choice=None) -> AgentExecutor:
    """
    Khởi tạo Language Model và Agent với Groq
    """
    # Khởi tạo language model với Groq
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        max_tokens=2000,
        model_kwargs={
            "messages": [
                {"role": "system", "content": VIETNAMESE_SYSTEM_PROMPT}
            ]
        }
    )
    
    tools = [tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", VIETNAMESE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Khởi tạo retriever và agent
retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)
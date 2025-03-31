# Import các thư viện cần thiết
import os
from dotenv import load_dotenv
import logging
from pydantic import BaseModel, Field

load_dotenv()

# Chỉ kiểm tra GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

from langchain.tools.retriever import create_retriever_tool  # Tạo công cụ tìm kiếm
from langchain_groq import ChatGroq  # Model ngôn ngữ Groq
from langchain.agents import AgentExecutor, create_openai_functions_agent  # Tạo và thực thi agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Xử lý prompt
from seed_data import connect_to_chromadb, get_available_collections  # Kết nối với ChromaDB và lấy danh sách collections
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler  # Xử lý callback cho Streamlit
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # Lưu trữ lịch sử chat
from langchain_core.documents import Document  # Lớp Document
from langchain_core.retrievers import BaseRetriever  # BaseRetriever interface

def determine_collection(query: str) -> str:
    """Xác định collection phù hợp với câu hỏi của người dùng"""
    query_lower = query.lower()
    
    # Kiểm tra từ khóa để xác định collection
    if any(keyword in query_lower for keyword in ["nước ngoài", "quốc tế", "ngoại quốc", "người nước ngoài"]):
        return "nuoc_ngoai"
    elif any(keyword in query_lower for keyword in ["liên thông", "kết hợp", "đồng thời"]):
        return "lien_thong"
    elif any(keyword in query_lower for keyword in ["lưu động", "đăng ký lại", "đã có hồ sơ", "đặc biệt"]):
        return "dac_biet"
    else:
        return "trong_nuoc"

class SmartRetriever(BaseRetriever, BaseModel):
    persist_dir: str = Field(default="chroma_db")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str):
        """Required abstract method implementation from BaseRetriever"""
        try:
            collection_name = determine_collection(query)
            logging.info(f"Tìm kiếm trong collection: {collection_name}")
            
            vectorstore = connect_to_chromadb(self.persist_dir, collection_name)
            results = vectorstore.similarity_search(query, k=4)
            logging.info(f"Tìm thấy {len(results)} kết quả từ {collection_name}")
            
            return results
            
        except Exception as e:
            logging.error(f"Lỗi khi tìm kiếm: {str(e)}")
            return [Document(
                page_content="Có lỗi xảy ra khi tìm kiếm. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )]
    
    def invoke(self, input_query: str, **kwargs):
        """New invoke method that calls _get_relevant_documents"""
        return self._get_relevant_documents(input_query)

def get_retriever(persist_dir: str = None) -> SmartRetriever:
    """
    Tạo retriever thông minh dựa trên nội dung câu hỏi
    Args:
        persist_dir: Đường dẫn tuyệt đối đến thư mục ChromaDB
    """
    if persist_dir is None:
        persist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    return SmartRetriever(persist_dir=persist_dir)

# Cập nhật tool với đường dẫn tuyệt đối
tool = create_retriever_tool(
    get_retriever(os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")),
    "find",
    "Tìm kiếm thông tin về thủ tục hành chính công dựa trên từ khóa trong câu hỏi."
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
    )
    
    llm = llm.bind(
        system_prompt=VIETNAMESE_SYSTEM_PROMPT
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

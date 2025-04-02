import os
from dotenv import load_dotenv
import pandas as pd
import openpyxl
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import asyncio
import nest_asyncio

VIETNAMESE_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên trả lời các câu hỏi về thủ tục hành chính công, đặc biệt là các thủ tục về hộ tịch như đăng ký khai sinh, khai tử, kết hôn.
Hãy trả lời các câu hỏi bằng tiếng Việt một cách ngắn gọn, rõ ràng theo từng điểm.
Đối với câu hỏi về giấy tờ cần chuẩn bị, hãy liệt kê theo dạng danh sách có đánh số.
Nếu không có thông tin, hãy thông báo rõ ràng là chưa có thông tin về vấn đề đó.
Luôn giữ giọng điệu lịch sự, chuyên nghiệp và nhắc nhở người dùng về các lưu ý quan trọng."""

# Thêm hàm ghi log
def log_to_excel(context, question, assistant_thought, answer, references, log_file="logs/chat_logs.xlsx"):
    os.makedirs("logs", exist_ok=True)
    
    # Lấy thời gian hiện tại và định dạng theo kiểu dd/mm/yyyy HH:MM:SS
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Chuẩn bị dữ liệu với định dạng thời gian đã được format
    data = {
        'Timestamp': [current_time],  # Đã được format
        'Context': [context],
        'Question': [question],
        'Assistant_Thought': [assistant_thought],
        'Answer': [answer],
        'References': [references]
    }
    
    # Tạo DataFrame mới
    new_df = pd.DataFrame(data)
    
    # Đọc file cũ nếu tồn tại
    if os.path.exists(log_file):
        existing_df = pd.read_excel(log_file)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    
    # Lưu vào file Excel với định dạng cột thời gian
    with pd.ExcelWriter(log_file, engine='openpyxl') as writer:
        updated_df.to_excel(writer, index=False)
        # Định dạng cột thời gian
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        timestamp_col = worksheet['A']
        for cell in timestamp_col[1:]:  # Bỏ qua header
            cell.number_format = 'dd/mm/yyyy hh:mm:ss'

# 1. Tải các vector database đã tạo
def load_retrievers(base_persist_directory):
    # Khởi tạo embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={"trust_remote_code": True}
    )
    
    # Danh sách các collection
    parent_collection = "Ho_Tich"
    categories = ["trong_nuoc", "nuoc_ngoai", "dac_biet", "lien_thong"]
    retrievers = {}
    vectordbs = {}
    
    # Tải từng collection và tạo retriever tương ứng
    for category in categories:
        persist_directory = os.path.join(base_persist_directory, parent_collection, category)
        if os.path.exists(persist_directory):
            db = Chroma(
                persist_directory=persist_directory, 
                embedding_function=embeddings
            )
            vectordbs[category] = db
            retrievers[category] = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 4, "score_threshold": 0.45}
            )
            print(f"Đã tải retriever cho collection '{category}'")
    
    return vectordbs, retrievers

# 2. Xác định collection phù hợp với câu hỏi
def determine_collection(query):
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

# 3. Tạo chatbot với khả năng chọn collection phù hợp
def build_rag_chatbots(vectordbs):
    # Load environment variables
    load_dotenv()
    
    try:
        # Enable nested event loops
        nest_asyncio.apply()
        
        # Create new event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Khởi tạo language model với Groq
        llm = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b",
            temperature=0.5,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=2000
        )
        
        # Test LLM with simple input
        test_response = llm.invoke("Xin chào")
        print("LLM test thành công:", test_response)

        # Tạo bộ nhớ cho cuộc trò chuyện
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        
        # Đơn giản hóa prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Dựa vào thông tin sau đây, hãy trả lời câu hỏi một cách trực tiếp bằng tiếng Việt.
Nếu không có thông tin liên quan, hãy trả lời "Tôi không tìm thấy thông tin về vấn đề này."

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""
        )
        
        # Dictionary để lưu trữ các chain cho từng collection
        qa_chains = {}
        
        # Tạo chain cho từng collection
        for category, vectordb in vectordbs.items():
            retriever = vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.4}
            )
            
            qa_chains[category] = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": prompt_template}
            )
        return qa_chains
        
    except Exception as e:
        print(f"Lỗi khi khởi tạo chatbot: {str(e)}")
        return None

# 4. Xử lý câu hỏi và trả lời
def process_query(query, qa_chains, retrievers):
    collection = determine_collection(query)
    
    if collection in qa_chains:
        docs = retrievers[collection].get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Thêm phần <think> cho assistant
        assistant_thought = f"<think>Dựa trên collection {collection} và {len(docs)} tài liệu tham khảo</think>"
        
        result = qa_chains[collection].invoke({"question": query})
        answer = result["answer"]
        
        # Xử lý lại chuỗi trả về
        if "<answer>" in answer:
            answer = answer.split("<answer>")[1].split("</answer>")[0].strip()
            answer = answer.replace("[Phần này sẽ hiển thị cho người dùng]", "").strip()
        
        # Tạo danh sách tài liệu tham khảo
        references = []
        for doc in result.get("source_documents", []):
            source_file = doc.metadata.get('source', 'Unknown')
            page_number = doc.metadata.get('page', 'Unknown')
            references.append(f"File: {os.path.basename(source_file)}, Trang: {page_number}")
        references_str = "\n".join(references)
        
        # Ghi log với format mới
        log_to_excel(context, query, assistant_thought, answer, references_str)
        
        return {
            "answer": answer,
            "collection": collection,
            "source_documents": result.get("source_documents", [])
        }
    else:
        return {
            "answer": "Xin lỗi, không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau.",
            "collection": None,
            "source_documents": []
        }

# 5. Tích hợp vào giao diện người dùng
def run_chatbot(qa_chains, retrievers):
    print("Chào mừng bạn đến với RAG Chatbot! Gõ 'exit' để thoát.")
    while True:
        query = input("\nBạn: ")
        if query.lower() == "exit":
            break

        # Xử lý câu hỏi
        result = process_query(query, qa_chains, retrievers)
        
        # Hiển thị kết quả
        print(f"\nBot [{result['collection']}]: {result['answer']}")
        
        # Hiển thị nguồn tài liệu
        sources = result.get("source_documents", [])
        if sources:
            print("\nNguồn tham khảo:")
            for i, doc in enumerate(sources):
                source_file = doc.metadata.get('source', 'Unknown')
                page_number = doc.metadata.get('page', 'Unknown')
                print(f"{i + 1}. File: {os.path.basename(source_file)}, Trang: {page_number}")

def main():
    # Thư mục gốc chứa các vector database
    base_persist_directory = r"E:\WORK\project\chatbot_RAG\chroma_db"
    
    # Tải các retriever
    vectordbs, retrievers = load_retrievers(base_persist_directory)
    
    # Kiểm tra xem có tải được vector database nào không
    if not vectordbs:
        print("Không tìm thấy collection nào. Vui lòng chạy create_vectordb.py trước.")
        return
    
    # Tạo chatbot
    qa_chains = build_rag_chatbots(vectordbs)
    
    # Kiểm tra xem qa_chains có được khởi tạo thành công không
    if qa_chains is None:
        print("Không thể khởi tạo chatbot. Vui lòng kiểm tra lại cấu hình và API key.")
        return
        
    # Chạy chatbot
    run_chatbot(qa_chains, retrievers)

if __name__ == "__main__":
    main()

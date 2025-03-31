import os
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def get_embeddings():
    """
    Khởi tạo HuggingFace Embeddings cho tiếng Việt
    """
    return HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={"trust_remote_code": True}
    )

def load_pdf_documents(pdf_directory):
    """
    Đọc tất cả file PDF từ thư mục
    """
    loader = DirectoryLoader(
        pdf_directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    return documents

def split_documents(documents):
    """
    Chia nhỏ tài liệu thành các đoạn
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def classify_document(document):
    """
    Phân loại tài liệu theo nội dung
    """
    source = document.metadata.get("source", "").lower()
    content = document.page_content.lower()
    
    if "yếu tố nước ngoài" in source or "nước ngoài" in source:
        return "nuoc_ngoai"
    elif "liên thông" in source:
        return "lien_thong"
    elif any(term in source for term in ["lưu động", "đăng ký lại", "người đã có hồ sơ", "thay đổi", "cải chính", "xác định lại", "bổ sung", "kết hợp"]):
        return "dac_biet"
    else:
        return "trong_nuoc"

def create_vectordbs(chunks, base_persist_directory):
    """
    Tạo và lưu vector embeddings cho từng loại tài liệu
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={"trust_remote_code": True}
    )
    
    categorized_chunks = {
        "trong_nuoc": [],
        "nuoc_ngoai": [],
        "dac_biet": [],
        "lien_thong": []
    }
    
    for chunk in chunks:
        category = classify_document(chunk)
        categorized_chunks[category].append(chunk)
    
    vectordbs = {}
    for category, category_chunks in categorized_chunks.items():
        if category_chunks:
            persist_directory = os.path.join(base_persist_directory, category)
            vectordbs[category] = Chroma.from_documents(
                documents=category_chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            print(f"Đã tạo vector database '{category}' với {len(category_chunks)} đoạn văn bản")
            
            files = set(chunk.metadata.get("source", "Không rõ") for chunk in category_chunks)
            print(f"Các file trong collection '{category}':")
            for file in files:
                print(f"- {file}")
    
    return vectordbs

def get_available_collections(base_persist_directory: str) -> list:
    """
    Lấy danh sách tất cả các collection có sẵn
    """
    collections = []
    categories = ["trong_nuoc", "nuoc_ngoai", "dac_biet", "lien_thong"]
    
    for category in categories:
        persist_directory = os.path.join(base_persist_directory, category)
        if os.path.exists(persist_directory):
            collections.append(category)
            
    return collections

def seed_pdf_data(pdf_directory: str, persist_directory: str):
    """
    Hàm chính để xử lý PDF và tạo vector database
    """
    # 1. Load PDF documents
    documents = load_pdf_documents(pdf_directory)
    print(f"Đã tải {len(documents)} tài liệu PDF")
    
    # 2. Split documents
    chunks = split_documents(documents)
    print(f"Đã chia thành {len(chunks)} đoạn văn bản")
    
    # 3. Create vector databases
    vectordbs = create_vectordbs(chunks, persist_directory)
    return vectordbs

def connect_to_chromadb(persist_directory: str, collection_name: str) -> Chroma:
    """
    Hàm kết nối đến collection có sẵn trong ChromaDB
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return vectorstore

def main():
    """
    Hàm chính để kiểm thử các chức năng của module
    """
    persist_dir = "chroma_db"
    pdf_dir = "E:/WORK/project/chatbot_RAG/data/pdf"  # Sửa dấu \ thành /
    seed_pdf_data(pdf_dir, persist_dir)
    
# Chạy main() nếu file được thực thi trực tiếp
if __name__ == "__main__":
    main()

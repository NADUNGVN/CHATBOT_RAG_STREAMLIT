import os
import shutil
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Tải và xử lý dữ liệu từ nhiều file PDF
def load_pdf_documents(pdf_directory):
    loader = DirectoryLoader(
        pdf_directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    return documents

# 2. Chia nhỏ tài liệu với kích thước phù hợp cho PDF
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Hàm phân loại tài liệu dựa trên tên file hoặc nội dung
def classify_document(document):
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

# 3. Tạo embeddings và lưu trữ vector cho từng collection
def create_vectordbs(chunks, base_persist_directory):
    # Sử dụng mô hình embedding từ Hugging Face (hỗ trợ tiếng Việt)
    embeddings = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={"trust_remote_code": True}
    )
    
    # Phân loại các đoạn văn bản
    categorized_chunks = {
        "trong_nuoc": [],
        "nuoc_ngoai": [],
        "dac_biet": [],
        "lien_thong": []
    }
    
    for chunk in chunks:
        category = classify_document(chunk)
        categorized_chunks[category].append(chunk)
    
    # Tạo vector database cho từng loại trong collection cha "Hộ Tịch"
    parent_collection = "Ho_Tich"
    vectordbs = {}
    parent_directory = os.path.join(base_persist_directory, parent_collection)
    os.makedirs(parent_directory, exist_ok=True)

    for category, category_chunks in categorized_chunks.items():
        if category_chunks:
            persist_directory = os.path.join(parent_directory, category)
            vectordbs[category] = Chroma.from_documents(
                documents=category_chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            print(f"Đã tạo vector database '{category}' trong collection cha '{parent_collection}' với {len(category_chunks)} đoạn văn bản")

            # Lấy danh sách file từ metadata
            files = set(chunk.metadata.get("source", "Không rõ") for chunk in category_chunks)
            print(f"Các file trong collection '{category}':")
            for file in files:
                print(f"- {file}")
    
    return vectordbs

# 4. Tải lại vector database đã có (để tránh phải xử lý lại PDF)
def load_vectordbs(base_persist_directory, embeddings):
    vectordbs = {}
    categories = ["trong_nuoc", "nuoc_ngoai", "dac_biet", "lien_thong"]
    
    for category in categories:
        persist_directory = os.path.join(base_persist_directory, category)
        if os.path.exists(persist_directory):
            vectordbs[category] = Chroma(
                persist_directory=persist_directory, 
                embedding_function=embeddings
            )
    
    return vectordbs

# Hàm chính
def main():
    pdf_directory = r"E:\WORK\project\chatbot_RAG\data\pdf"
    base_persist_directory = r"E:\WORK\project\chatbot_RAG\chroma_db"
    
    if os.path.exists(base_persist_directory):
        shutil.rmtree(base_persist_directory)
    
    os.makedirs(base_persist_directory, exist_ok=True)
    
    documents = load_pdf_documents(pdf_directory)
    chunks = split_documents(documents)
    vectordbs = create_vectordbs(chunks, base_persist_directory)

if __name__ == "__main__":
    main()

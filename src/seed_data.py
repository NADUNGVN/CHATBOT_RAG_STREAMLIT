import os
import json
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
from crawl import crawl_web
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def get_embeddings():
    """
    Khởi tạo HuggingFace Embeddings cho tiếng Việt
    """
    return HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={"trust_remote_code": True}
    )

def load_data_from_local(filename: str, directory: str) -> tuple:
    """
    Hàm đọc dữ liệu từ file JSON local
    Args:
        filename (str): Tên file JSON cần đọc (ví dụ: 'data.json')
        directory (str): Thư mục chứa file (ví dụ: 'data_v3')
    Returns:
        tuple: Trả về (data, doc_name) trong đó:
            - data: Dữ liệu JSON đã được parse
            - doc_name: Tên tài liệu đã được xử lý (bỏ đuôi .json và thay '_' bằng khoảng trắng)
    """
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(f'Data loaded from {file_path}')
    # Chuyển tên file thành tên tài liệu (bỏ đuôi .json và thay '_' bằng khoảng trắng)
    return data, filename.rsplit('.', 1)[0].replace('_', ' ')

def seed_chromadb(persist_directory: str, collection_name: str, filename: str, directory: str, use_huggingface: bool = True) -> Chroma:
    """
    Hàm tạo và lưu vector embeddings vào ChromaDB từ dữ liệu local
    """
    embeddings = get_embeddings()
    
    local_data, doc_name = load_data_from_local(filename, directory)

    documents = [
        Document(
            page_content=doc.get('page_content') or '',
            metadata={
                'source': doc['metadata'].get('source') or '',
                'content_type': doc['metadata'].get('content_type') or 'text/plain',
                'title': doc['metadata'].get('title') or '',
                'description': doc['metadata'].get('description') or '',
                'language': doc['metadata'].get('language') or 'en',
                'doc_name': doc_name,
                'start_index': doc['metadata'].get('start_index') or 0
            }
        )
        for doc in local_data
    ]

    # Khởi tạo ChromaDB
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    # Thêm documents vào ChromaDB
    vectorstore.add_documents(documents=documents)
    
    # Lưu xuống ổ cứng
    vectorstore.persist()
    
    return vectorstore

def seed_chromadb_live(URL: str, persist_directory: str, collection_name: str, doc_name: str, use_huggingface: bool = True) -> Chroma:
    """
    Hàm crawl dữ liệu trực tiếp từ URL và tạo vector embeddings trong ChromaDB
    """
    embeddings = get_embeddings()
    
    documents = crawl_web(URL)

    for doc in documents:
        metadata = {
            'source': doc.metadata.get('source') or '',
            'content_type': doc.metadata.get('content_type') or 'text/plain',
            'title': doc.metadata.get('title') or '',
            'description': doc.metadata.get('description') or '',
            'language': doc.metadata.get('language') or 'en',
            'doc_name': doc_name,
            'start_index': doc.metadata.get('start_index') or 0
        }
        doc.metadata = metadata

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    vectorstore.add_documents(documents=documents)
    vectorstore.persist()
    return vectorstore

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
    Thực hiện:
        1. Test seed_chromadb với dữ liệu từ file local 'stack.json'
        2. (Đã comment) Test seed_chromadb_live với dữ liệu từ trang web stack-ai
    Chú ý:
        - Đảm bảo ChromaDB đã được cấu hình đúng
        - Các biến môi trường cần thiết (như OPENAI_API_KEY) đã được cấu hình
    """
    # Test seed_chromadb với dữ liệu local
    persist_dir = "chroma_db"
    seed_chromadb(persist_dir, 'data_test', 'stack.json', 'data', use_huggingface=True)
    # Test seed_chromadb_live với URL trực tiếp
    # seed_chromadb_live('https://www.stack-ai.com/docs', persist_dir, 'data_test_live', 'stack-ai', use_huggingface=True)

# Chạy main() nếu file được thực thi trực tiếp
if __name__ == "__main__":
    main()
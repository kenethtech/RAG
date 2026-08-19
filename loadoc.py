from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

PATH = "data"
CHROMA_PATH = "chroma"
embeddings= HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def load_documents():
    loader = DirectoryLoader(PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True
    )
    chunk = text_splitter.split_documents(documents)
    return chunk


def store_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    Chroma.from_documents(
        documents= chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

def content_retriever(queries):
    existing_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    result = existing_db.similarity_search_with_relevance_scores(queries, k=5)
    return result

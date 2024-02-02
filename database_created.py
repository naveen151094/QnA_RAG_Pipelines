from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os,shutil

DOC_PATH = "pdf_data"
CHROMA_PATH = "chroma"

def load_documents():
    loader = DirectoryLoader(DOC_PATH, glob = "*.pdf")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=120,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    vectordb = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory = CHROMA_PATH
    )
    vectordb.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")
    return vectordb

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    vectordb = save_to_chroma(chunks)
    return vectordb
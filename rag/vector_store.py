from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from config.settings import CHROMA_DIR

def get_vector_store():
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def add_documents(chunks):
    db = get_vector_store()
    db.add_documents(chunks)
from rag.loaders import load_sec_pdf
from rag.processors import chunk_documents
from rag.vector_store import add_documents
from loguru import logger

def ingest_sec_pdf(pdf_path: str):
    logger.info(f"Ingesting PDF: {pdf_path}")
    documents = load_sec_pdf(pdf_path)
    chunks = chunk_documents(documents)
    add_documents(chunks)
    logger.success(f"Ingested {len(chunks)} chunks")
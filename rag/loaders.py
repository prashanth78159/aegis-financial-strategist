from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_sec_pdf(pdf_path: str):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    return loader.load()
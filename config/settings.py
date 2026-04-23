import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ✅ MUST be local disk in Colab
CHROMA_DIR = "/content/chroma_afs"
RAW_DATA_DIR = "data/raw"
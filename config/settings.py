import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("AIzaSyBg60yNywN3i2RgoNr8FLa7JPEg5Eg1ulc")
TAVILY_API_KEY = os.getenv("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")

# IMPORTANT: Local disk (writable)
CHROMA_DIR = "/content/chroma_afs"
RAW_DATA_DIR = "data/raw"
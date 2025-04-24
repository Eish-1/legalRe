import os
from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter # Replace this
from langchain.text_splitter import RecursiveCharacterTextSplitter # Use this instead
from langchain_groq import ChatGroq
import logging
# ... rest of imports ...

logger = logging.getLogger('legalre')

def get_text_chunks(text):
    """Splits text into chunks using RecursiveCharacterTextSplitter."""
    if not text:
        return []
    # Switch to RecursiveCharacterTextSplitter for summarization chunks too
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # Slightly smaller than RAG chunk size
        chunk_overlap=100,
        length_function=len,
        add_start_index=True, # Good practice
        separators=["\n\n", "\n", ".", " ", ""] # Standard separators
    )
    # Use split_text for raw text input
    chunks = text_splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks for summarization.")
    return chunks

# ... generate_summary function remains the same ... 
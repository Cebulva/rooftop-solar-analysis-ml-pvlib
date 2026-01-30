"""
Complete RAG Setup Script for Solar Chatbot
============================================

Location: src/setup_rag.py

This script:
1. Extracts text from 5 PDF documents
2. Splits text into chunks (800 chars, 200 overlap)
3. Generates embeddings for each chunk
4. Stores in ChromaDB vector database

Run once to set up the RAG system:
    cd src/
    python3 setup_rag.py

Database will be created in: src/solar_rag_db/
"""

import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import sys
from typing import List, Dict
import os

# ============================================
# CONFIGURATION
# ============================================

CHUNK_SIZE = 800          # Characters per chunk
CHUNK_OVERLAP = 200       # Overlap between chunks
EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'  # German + English support

# Paths relative to src/ folder
SCRIPT_DIR = Path(__file__).parent
DB_PATH = str(SCRIPT_DIR.parent / "data" / "solar_rag_db")  # ../data/solar_rag_db
COLLECTION_NAME = "solar_regulations"

# PDF file paths (try multiple locations)
def find_uploads_dir():
    """Find the uploads directory"""
    possible_paths = [
        SCRIPT_DIR.parent / "data" / "uploads",  # ../data/uploads
        Path("/mnt/user-data/uploads"),           # Absolute path
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return Path("/mnt/user-data/uploads")  # Default fallback

UPLOADS_DIR = find_uploads_dir()

PDF_FILES = {
    "EEG_2023": str(UPLOADS_DIR / "EEG_2023.pdf"),
    "Tax_Exemption": str(UPLOADS_DIR / "2023-07-17-Photovoltaikanlagen-Steuerbefreiung.pdf"),
    "Consumer_FAQ": str(UPLOADS_DIR / "ihre-photovoltaikanlage.pdf"),
    "KfW_270": str(UPLOADS_DIR / "6000000178_M_270_EE-Standard.pdf"),
    "EEG_Rates_2025_2026": str(UPLOADS_DIR / "EEG_2025_2026 rates.pdf")
}

# ============================================
# STEP 1: TEXT EXTRACTION
# ============================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {i+1}]\n{page_text}\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# ============================================
# STEP 2: TEXT CHUNKING
# ============================================

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, 
                          overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('. ')
            last_newline = chunk.rfind('\n')
            last_space = chunk.rfind(' ', -100)
            
            break_point = max(last_period, last_newline, last_space)
            if break_point > chunk_size * 0.7:
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text) - overlap:
            break
    
    return chunks

# ============================================
# STEP 3: PROCESS ALL PDFs
# ============================================

def process_all_pdfs() -> List[Dict]:
    """Extract text from all PDFs and create chunks."""
    all_chunks = []
    
    print("\n" + "="*60)
    print("STEP 1-2: EXTRACTING AND CHUNKING PDFs")
    print("="*60 + "\n")
    
    for doc_name, pdf_path in PDF_FILES.items():
        print(f"Processing {doc_name}...")
        
        if not Path(pdf_path).exists():
            print(f"  ⚠️  File not found: {pdf_path}")
            continue
        
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text:
            print(f"  ⚠️  No text extracted")
            continue
        
        print(f"  Extracted: {len(full_text):,} characters")
        
        chunks = split_text_into_chunks(full_text)
        print(f"  Created: {len(chunks)} chunks")
        
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "doc_name": doc_name,
                "chunk_id": f"{doc_name}_chunk_{i:03d}",
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        print(f"  ✓ Completed\n")
    
    print(f"Total chunks created: {len(all_chunks)}\n")
    return all_chunks

# ============================================
# STEP 4: GENERATE EMBEDDINGS & STORE
# ============================================

def setup_vector_database(chunks: List[Dict]):
    """Generate embeddings and store in ChromaDB."""
    print("="*60)
    print("STEP 3-4: GENERATING EMBEDDINGS & STORING")
    print("="*60 + "\n")
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Loaded: {EMBEDDING_MODEL}\n")
    
    print("Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection")
    except:
        pass
    
    collection = chroma_client.create_collection(COLLECTION_NAME)
    print(f"✓ Created collection: {COLLECTION_NAME}\n")
    
    print("Embedding and storing chunks...")
    print("(This may take 1-2 minutes)\n")
    
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        
        ids = [chunk["chunk_id"] for chunk in batch]
        metadatas = [{
            "doc_name": chunk["doc_name"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"]
        } for chunk in batch]
        
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        print(f"  Processed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks", end='\r')
    
    print(f"\n✓ Stored {len(chunks)} chunks in vector database\n")
    return collection

# ============================================
# STEP 5: TEST RETRIEVAL
# ============================================

def test_retrieval():
    """Test the RAG system with sample queries."""
    print("="*60)
    print("STEP 5: TESTING RETRIEVAL")
    print("="*60 + "\n")
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    
    test_queries = [
        "What is the feed-in tariff for systems up to 10 kWp?",
        "Can I get BEG subsidy for my solar panels?",
        "What is the tax exemption limit for 2025?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 60)
        
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            doc_name = metadata['doc_name']
            chunk_idx = metadata['chunk_index']
            print(f"\nResult {i+1}: {doc_name} (chunk {chunk_idx})")
            print(f"Text preview: {doc[:200]}...")
        
        print("\n" + "="*60 + "\n")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("SOLAR CHATBOT RAG SETUP")
    print("="*60)
    
    try:
        chunks = process_all_pdfs()
        
        if not chunks:
            print("❌ No chunks created. Check if PDF files exist.")
            return
        
        collection = setup_vector_database(chunks)
        test_retrieval()
        
        print("="*60)
        print("✓ RAG SETUP COMPLETE!")
        print("="*60)
        print(f"\nDatabase location: {DB_PATH}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Collection name: {COLLECTION_NAME}")
        print("\nNext: Import retrieval functions from rag_retrieval.py in your chatbot")
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
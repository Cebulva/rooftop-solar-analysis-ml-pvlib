"""
RAG Retrieval Functions for Solar Chatbot
==========================================

Location: src/rag_retrieval.py

Import this in your rag_bot.py:
    from rag_retrieval import retrieve_relevant_chunks

Use in your message handler:
    rag_context = retrieve_relevant_chunks(user_message)
    if rag_context:
        system_prompt += f"\\n\\n=== DETAILED REGULATIONS ===\\n{rag_context}"
"""

from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
from typing import Optional

# ============================================
# CONFIGURATION
# ============================================

EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
SCRIPT_DIR = Path(__file__).parent
DB_PATH = str(SCRIPT_DIR.parent / "data" / "solar_rag_db")  # ../data/solar_rag_db
COLLECTION_NAME = "solar_regulations"

# Cache the model and client (loaded once, reused for all queries)
_embedding_model = None
_chroma_client = None
_collection = None

# ============================================
# INITIALIZATION
# ============================================

def initialize_rag():
    """
    Initialize RAG system (load model and connect to database).
    Called automatically on first retrieval.
    """
    global _embedding_model, _chroma_client, _collection
    
    if _embedding_model is None:
        print("Loading embedding model (first time only)...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✓ Model loaded")
    
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _chroma_client.get_collection(COLLECTION_NAME)

# ============================================
# RETRIEVAL FUNCTIONS
# ============================================

def retrieve_relevant_chunks(query: str, n_results: int = 2) -> Optional[str]:
    """
    Retrieve relevant document chunks for a user query.
    
    Args:
        query: User's question
        n_results: Number of chunks to retrieve (default: 2)
        
    Returns:
        Formatted string with retrieved chunks and sources
        Returns None if no results or database not available
        
    Example:
        context = retrieve_relevant_chunks("What is the feed-in tariff?")
        if context:
            system_prompt += f"\\n\\n=== DETAILED REGULATIONS ===\\n{context}"
    """
    try:
        # Initialize on first call
        initialize_rag()
        
        # Embed the query
        query_embedding = _embedding_model.encode(query).tolist()
        
        # Retrieve similar chunks
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return None
        
        # Format results with sources
        context = ""
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            doc_name = metadata['doc_name']
            context += f"\\n=== Source: {doc_name} ===\\n{doc}\\n"
        
        return context.strip()
        
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return None

def should_use_rag(query: str) -> bool:
    """
    Determine if RAG should be used for this query.
    
    Args:
        query: User's question
        
    Returns:
        True if RAG should be used, False otherwise
        
    Use this to avoid RAG for simple questions:
        if should_use_rag(user_message):
            rag_context = retrieve_relevant_chunks(user_message)
    """
    # Keywords that indicate need for detailed information
    detail_keywords = [
        "details", "more info", "explain", "how does", "what are all",
        "conditions", "requirements", "exactly", "specific",
        "mehr info", "genau", "bedingungen", "voraussetzungen",
        "§", "paragraph", "law", "gesetz", "regulation"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in detail_keywords)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def check_rag_available() -> bool:
    """
    Check if RAG database is available.
    
    Returns:
        True if database exists and can be accessed
    """
    try:
        db_path = Path(DB_PATH)
        if not db_path.exists():
            return False
        
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        return True
    except:
        return False

def get_rag_stats() -> dict:
    """
    Get statistics about the RAG database.
    
    Returns:
        Dictionary with stats (total_chunks, collection_name, etc.)
    """
    try:
        initialize_rag()
        count = _collection.count()
        
        return {
            "available": True,
            "total_chunks": count,
            "collection_name": COLLECTION_NAME,
            "db_path": DB_PATH,
            "model": EMBEDDING_MODEL
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """Test the retrieval functions"""
    
    print("Testing RAG retrieval functions...")
    print("="*60)
    
    # Check if RAG is available
    if not check_rag_available():
        print("❌ RAG database not found!")
        print("Run setup_rag.py first to create the database.")
        exit(1)
    
    # Get stats
    stats = get_rag_stats()
    print(f"\\nRAG Stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Database: {stats['db_path']}")
    
    # Test queries
    test_queries = [
        "What is the feed-in tariff?",
        "Can I get BEG for solar?",
        "Tax exemption limit?"
    ]
    
    print(f"\\nTesting {len(test_queries)} queries...")
    print("="*60)
    
    for query in test_queries:
        print(f"\\nQuery: {query}")
        print(f"Should use RAG: {should_use_rag(query)}")
        
        context = retrieve_relevant_chunks(query, n_results=1)
        if context:
            print(f"Retrieved: {len(context)} characters")
            print(f"Preview: {context[:150]}...")
        else:
            print("No results")
        print("-"*60)
    
    print("\\n✓ Test complete!")
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    chroma_dirs = [d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith("chroma")]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))

            collections = client.list_collections()

            for collection in collections:
                key = f"{chroma_dir.name}::{collection.name}"
                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = 0
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} / {collection.name} ({doc_count} docs)",
                    "doc_count": doc_count
                }

        except Exception as e:
            key = f"{chroma_dir.name}::error"
            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (error: {str(e)[:50]})",
                "doc_count": 0
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(name=collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None

    if mission_filter and mission_filter.lower() not in ("all", ""):
        where_filter = {"mission": {"$eq": mission_filter}}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    context_parts = ["=== RETRIEVED DOCUMENTS ==="]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        mission = meta.get("mission", "unknown").replace("_", " ").title()
        source = meta.get("source", "unknown")
        category = meta.get("document_category", "unknown").replace("_", " ").title()

        header = f"\n--- Source {i}: {mission} | {source} | {category} ---"
        context_parts.append(header)

        max_chunk = 1500
        context_parts.append(doc[:max_chunk] + "..." if len(doc) > max_chunk else doc)

    return "\n".join(context_parts)
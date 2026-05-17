import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith("chroma")]

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(path=str(chroma_dir))

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            # Loop through each collection found
            for collection in collections:
                # Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}::{collection.name}"
                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = 0
                # Build information dictionary and add collection information to backends dictionary
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} / {collection.name} ({doc_count} docs)",
                    "doc_count": doc_count
                }

        # Handle connection or access errors gracefully
        except Exception as e:
            # Create fallback entry for inaccessible directories
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
        # Create a ChromaDB PersistentClient
        client = chromadb.PersistentClient(path=chroma_dir)
        # Return the collection with the collection_name
        collection = client.get_collection(name=collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    where_filter = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() not in ("all", ""):
        where_filter = {"mission": {"$eq": mission_filter}}

    # Execute database query
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    # Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # Initialize list with header text for context section
    context_parts = ["=== RETRIEVED DOCUMENTS ==="]

    # Loop through paired documents and their metadata using enumeration
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        # Extract mission, source, category information from metadata with fallback values
        mission = meta.get("mission", "unknown").replace("_", " ").title()
        source = meta.get("source", "unknown")
        category = meta.get("document_category", "unknown").replace("_", " ").title()

        # Create formatted source header with index number and extracted information
        header = f"\n--- Source {i}: {mission} | {source} | {category} ---"
        context_parts.append(header)

        # Check document length and truncate if necessary
        max_chunk = 1500
        context_parts.append(doc[:max_chunk] + "..." if len(doc) > max_chunk else doc)

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)
import os
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
from config import vector_store
from embeddingUtils import generate_embeddings

# Try to import Faiss, with fallback if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Faiss not available. Using fallback implementation.")

# Global variables for Faiss index and storage
faiss_index = None
faiss_chunks = []
dimension = 768  # Default dimension, will be updated when adding embeddings

def initialize_faiss_index(dim: int = 768) -> None:
    """
    Initialize the Faiss index with the specified dimension.
    
    Args:
        dim (int): The dimension of the embeddings. Default is 768.
    """
    global faiss_index, dimension
    if not FAISS_AVAILABLE:
        print("Faiss not available. Cannot initialize index.")
        return
        
    dimension = dim
    # Using IndexFlatIP for inner product (cosine similarity)
    # Normalize vectors to unit length for proper cosine similarity
    faiss_index = faiss.IndexFlatIP(dim)

def add_to_faiss_store(embeddings: np.ndarray, chunks: List[str]) -> None:
    """
    Add embeddings and their corresponding text chunks to the Faiss store.

    Args:
        embeddings (np.ndarray): A NumPy array containing the embeddings to add.
        chunks (List[str]): A list of text chunks corresponding to the embeddings.

    Returns:
        None
    """
    global faiss_index, faiss_chunks, dimension
    
    if not FAISS_AVAILABLE:
        print("Faiss not available. Cannot add to store.")
        return
    
    # Ensure embeddings are float32
    embeddings = embeddings.astype('float32')
    
    # Initialize index if it hasn't been created yet
    if faiss_index is None:
        if embeddings.shape[0] > 0:
            dimension = embeddings.shape[1]
        initialize_faiss_index(dimension)
    
    # Normalize embeddings to unit length for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to the Faiss index
    faiss_index.add(embeddings)
    
    # Store the corresponding text chunks
    faiss_chunks.extend(chunks)

def similarity_search_faiss(query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
    """
    Perform similarity search using Faiss and return the top_k most similar chunks.

    Args:
        query_embedding (np.ndarray): The embedding vector of the query.
        top_k (int): The number of most similar chunks to retrieve. Default is 5.

    Returns:
        List[str]: A list of the top_k most similar text chunks.
    """
    global faiss_index, faiss_chunks
    
    if not FAISS_AVAILABLE:
        print("Faiss not available. Cannot perform similarity search.")
        return []
    
    if faiss_index is None or faiss_index.ntotal == 0:
        return []
    
    # Normalize query embedding to unit length
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Perform search using Faiss
    distances, indices = faiss_index.search(query_embedding, min(top_k, faiss_index.ntotal))
    
    # Retrieve the corresponding text chunks
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < len(faiss_chunks):
            results.append(faiss_chunks[idx])
    
    return results

def retrieve_relevant_chunks_faiss(query_text: str, top_k: int = 5) -> List[str]:
    """
    Retrieve the most relevant document chunks for a given query text using Faiss.

    Args:
        query_text (str): The query text for which relevant chunks are to be retrieved.
        top_k (int): The number of most relevant chunks to retrieve. Default is 5.

    Returns:
        List[str]: A list of the top_k most relevant text chunks.
    """
    if not FAISS_AVAILABLE:
        print("Faiss not available. Cannot retrieve relevant chunks.")
        return []
        
    # Generate embedding for the query text using the embedding model
    query_embedding = generate_embeddings([query_text])[0]
    
    # Ensure the query embedding has the correct dimension
    global dimension
    if len(query_embedding) != dimension:
        # If dimensions don't match, we need to handle this appropriately
        # For now, we'll print a warning and truncate/pad as needed
        print(f"Warning: Query embedding dimension ({len(query_embedding)}) does not match index dimension ({dimension})")
        if len(query_embedding) > dimension:
            query_embedding = query_embedding[:dimension]
        else:
            # Pad with zeros if embedding is too short
            padded = np.zeros(dimension)
            padded[:len(query_embedding)] = query_embedding
            query_embedding = padded
    
    # Perform similarity search to find the most relevant chunks
    relevant_chunks = similarity_search_faiss(query_embedding, top_k=top_k)
    
    # Return the list of relevant chunks
    return relevant_chunks

def save_faiss_index(file_path: str) -> None:
    """
    Save the Faiss index and chunks to disk.

    Args:
        file_path (str): The path where the index should be saved (without extension).

    Returns:
        None
    """
    global faiss_index, faiss_chunks
    
    if not FAISS_AVAILABLE:
        print("Faiss not available. Cannot save index.")
        return
        
    if faiss_index is not None:
        # Save the Faiss index
        faiss.write_index(faiss_index, f"{file_path}.index")
        
        # Save the chunks to a JSON file
        with open(f"{file_path}_chunks.json", 'w', encoding='utf-8') as f:
            json.dump(faiss_chunks, f, ensure_ascii=False, indent=2)

def load_faiss_index(file_path: str) -> None:
    """
    Load the Faiss index and chunks from disk.

    Args:
        file_path (str): The path from where the index should be loaded (without extension).

    Returns:
        None
    """
    global faiss_index, faiss_chunks, dimension
    
    if not FAISS_AVAILABLE:
        print("Faiss not available. Cannot load index.")
        return
    
    # Load the Faiss index
    faiss_index = faiss.read_index(f"{file_path}.index")
    dimension = faiss_index.d
    
    # Load the chunks from the JSON file
    with open(f"{file_path}_chunks.json", 'r', encoding='utf-8') as f:
        faiss_chunks = json.load(f)
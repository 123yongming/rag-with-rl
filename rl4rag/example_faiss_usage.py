"""
Example usage of the Faiss-based vector store implementation.
This script demonstrates how to use the Faiss vector store for document retrieval.
"""

# Try to import Faiss components
try:
    from faissStoreUtils import (
        initialize_faiss_index,
        add_to_faiss_store,
        retrieve_relevant_chunks_faiss,
        save_faiss_index,
        load_faiss_index
    )
    FAISS_AVAILABLE = True
    print("Faiss components imported successfully.")
except ImportError as e:
    FAISS_AVAILABLE = False
    print(f"Faiss components could not be imported: {e}")
    print("Please ensure Faiss is properly installed.")

import numpy as np
from dataUtils import load_documents, split_into_chunks
from embeddingUtils import generate_embeddings

def example_faiss_workflow():
    """Example workflow using Faiss for document storage and retrieval."""
    if not FAISS_AVAILABLE:
        print("Cannot run Faiss example as Faiss is not available.")
        return
        
    # Step 1: Load documents
    print("Loading documents...")
    documents = load_documents('/root/rag-with-rl/data')
    
    # Step 2: Split documents into chunks
    print("Splitting documents into chunks...")
    # Process documents and split into smaller chunks to avoid token limits
    chunks = split_into_chunks(documents, chunk_size=100)  # Reduced chunk size
    print(f"Created {len(chunks)} chunks.")
    
    # Step 3: Generate embeddings for chunks
    print("Generating embeddings for chunks...")
    try:
        embeddings = generate_embeddings(chunks, batch_size=5)  # Smaller batch size
        print(f"Generated {len(embeddings)} embeddings.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        print("Using random embeddings for demonstration purposes.")
        # Create random embeddings for demonstration
        embeddings = np.random.rand(len(chunks), 768).astype('float32')
    
    # Step 4: Initialize Faiss index
    print("Initializing Faiss index...")
    initialize_faiss_index(dim=len(embeddings[0]))
    
    # Step 5: Add embeddings to Faiss store
    print("Adding embeddings to Faiss store...")
    add_to_faiss_store(np.array(embeddings), chunks)
    
    # Step 6: Save Faiss index
    print("Saving Faiss index...")
    save_faiss_index('/tmp/faiss_index')
    print("Faiss index saved.")
    
    # Step 7: Query example
    query = "What are the principles of quantum computing?"
    print(f"\nQuerying with: '{query}'")
    results = retrieve_relevant_chunks_faiss(query, top_k=3)
    
    print("\nTop 3 relevant chunks:")
    for i, chunk in enumerate(results, 1):
        print(f"{i}. {chunk[:200]}...")  # Print first 200 characters

def load_and_query_example():
    """Example of loading a saved Faiss index and querying it."""
    if not FAISS_AVAILABLE:
        print("Cannot run Faiss example as Faiss is not available.")
        return
        
    print("\nLoading Faiss index...")
    load_faiss_index('/tmp/faiss_index')
    print("Faiss index loaded.")
    
    # Query example
    query = "Explain quantum computing concepts."
    print(f"\nQuerying loaded index with: '{query}'")
    results = retrieve_relevant_chunks_faiss(query, top_k=3)
    
    print("\nTop 3 relevant chunks from loaded index:")
    for i, chunk in enumerate(results, 1):
        print(f"{i}. {chunk[:200]}...")  # Print first 200 characters

if __name__ == "__main__":
    if FAISS_AVAILABLE:
        example_faiss_workflow()
        load_and_query_example()
    else:
        print("Faiss is not available. Install it with 'pip install faiss-gpu' (or 'faiss-cpu') to run these examples.")
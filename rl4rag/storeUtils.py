import os
from openai import OpenAI
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
from config import vector_store
from embeddingUtils import generate_embeddings

# Function to add embeddings and corresponding text chunks to the vector store
def add_to_vector_store(embeddings: np.ndarray, chunks: List[str]) -> None:
    """
    Add embeddings and their corresponding text chunks to the vector store.

    Args:
        embeddings (np.ndarray): A NumPy array containing the embeddings to add.
        chunks (List[str]): A list of text chunks corresponding to the embeddings.

    Returns:
        None
    """
    # Iterate over embeddings and chunks simultaneously
    for embedding, chunk in zip(embeddings, chunks):
        # Add each embedding and its corresponding chunk to the vector store
        # Use the current length of the vector store as the unique key
        vector_store[len(vector_store)] = {"embedding": embedding, "chunk": chunk}
    

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors, ranging from -1 to 1.
    """
    # Compute the dot product of the two vectors
    dot_product = np.dot(vec1, vec2)
    # Compute the magnitude (norm) of the first vector
    norm_vec1 = np.linalg.norm(vec1)
    # Compute the magnitude (norm) of the second vector
    norm_vec2 = np.linalg.norm(vec2)
    # Return the cosine similarity as the ratio of the dot product to the product of the norms
    return dot_product / (norm_vec1 * norm_vec2)

# Function to perform similarity search in the vector store
def similarity_search(query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
    """
    Perform similarity search in the vector store and return the top_k most similar chunks.

    Args:
        query_embedding (np.ndarray): The embedding vector of the query.
        top_k (int): The number of most similar chunks to retrieve. Default is 5.

    Returns:
        List[str]: A list of the top_k most similar text chunks.
    """
    similarities = []  # Initialize a list to store similarity scores and corresponding keys

    # Iterate through all items in the vector store
    for key, value in vector_store.items():
        # Compute the cosine similarity between the query embedding and the stored embedding
        similarity = cosine_similarity(query_embedding, value["embedding"])
        # Append the key and similarity score as a tuple to the list
        similarities.append((key, similarity))

    # Sort the list of similarities in descending order based on the similarity score
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Retrieve the top_k most similar chunks based on their keys
    return [vector_store[key]["chunk"] for key, _ in similarities[:top_k]]


# Function to retrieve relevant document chunks for a query
def retrieve_relevant_chunks(query_text: str, top_k: int = 5) -> List[str]:
    """
    Retrieve the most relevant document chunks for a given query text.

    Args:
        query_text (str): The query text for which relevant chunks are to be retrieved.
        top_k (int): The number of most relevant chunks to retrieve. Default is 5.

    Returns:
        List[str]: A list of the top_k most relevant text chunks.
    """
    # Generate embedding for the query text using the embedding model
    query_embedding = generate_embeddings([query_text])[0]
    
    # Perform similarity search to find the most relevant chunks
    relevant_chunks = similarity_search(query_embedding, top_k=top_k)
    
    # Return the list of relevant chunks
    return relevant_chunks
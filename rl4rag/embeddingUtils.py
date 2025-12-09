import os
from openai import OpenAI
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
from config import client

# Function to generate embeddings for a single batch of text chunks
def generate_embeddings_batch(chunks_batch: List[str], model: str = "BAAI/bge-large-zh-v1.5") -> List[List[float]]:
    """
    Generate embeddings for a batch of text chunks using the OpenAI client.

    Args:
        chunks_batch (List[str]): A batch of text chunks to generate embeddings for.
        model (str): The model to use for embedding generation. Default is "BAAI/bge-en-icl".

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floats.
    """
    # Use the OpenAI client to create embeddings for the input batch
    response = client.embeddings.create(
        model=model,  # Specify the model to use for embedding generation
        input=chunks_batch  # Provide the batch of text chunks as input
    )
    # Extract embeddings from the response and return them
    embeddings = [item.embedding for item in response.data]
    return embeddings


# Function to generate embeddings for all chunks with batching
def generate_embeddings(chunks: List[str], batch_size: int = 10) -> np.ndarray:
    """
    Generate embeddings for all text chunks in batches.

    Args:
        chunks (List[str]): A list of text chunks to generate embeddings for.
        batch_size (int): The number of chunks to process in each batch. Default is 10.

    Returns:
        np.ndarray: A NumPy array containing embeddings for all chunks.
    """
    all_embeddings = []  # Initialize an empty list to store all embeddings

    # Iterate through the chunks in batches
    for i in range(0, len(chunks), batch_size):
        # Extract the current batch of chunks
        batch = chunks[i:i + batch_size]
        # Generate embeddings for the current batch
        embeddings = generate_embeddings_batch(batch)
        # Extend the list of all embeddings with the embeddings from the current batch
        all_embeddings.extend(embeddings)

    # Convert the list of embeddings to a NumPy array and return it
    return np.array(all_embeddings)

# Function to save embeddings to a file
def save_embeddings(embeddings: np.ndarray, output_file: str) -> None:
    """
    Save embeddings to a JSON file.

    Args:
        embeddings (np.ndarray): A NumPy array containing the embeddings to save.
        output_file (str): The path to the output JSON file where embeddings will be saved.

    Returns:
        None
    """
    # Open the specified file in write mode with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as file:
        # Convert the NumPy array to a list and save it as JSON
        json.dump(embeddings.tolist(), file)
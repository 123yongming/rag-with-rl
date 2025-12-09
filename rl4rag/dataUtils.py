import os
from openai import OpenAI
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union


# Function to load documents from a directory
def load_documents(directory_path: str) -> List[str]:
    """
    Load all text documents from the specified directory.

    Args:
        directory_path (str): Path to the directory containing text files.

    Returns:
        List[str]: A list of strings, where each string is the content of a text file.
    """
    documents = []  # Initialize an empty list to store document contents
    for filename in os.listdir(directory_path):  # Iterate through all files in the directory
        if filename.endswith(".txt"):  # Check if the file has a .txt extension
            # Open the file in read mode with UTF-8 encoding and append its content to the list
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents  # Return the list of document contents



    # Function to split documents into chunks
def split_into_chunks(documents: List[str], chunk_size: int = 30) -> List[str]:
    """
    Split documents into smaller chunks of specified size.

    Args:
        documents (List[str]): A list of document strings to be split into chunks.
        chunk_size (int): The maximum number of words in each chunk. Default is 100.

    Returns:
        List[str]: A list of chunks, where each chunk is a string containing up to `chunk_size` words.
    """
    chunks = []  # Initialize an empty list to store the chunks
    for doc in documents:  # Iterate through each document
        words = doc.split()  # Split the document into words
        # Create chunks of the specified size
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])  # Join words to form a chunk
            chunks.append(chunk)  # Add the chunk to the list
    return chunks  # Return the list of chunks


    # Function to preprocess text (e.g., lowercasing, removing special characters)
def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by converting it to lowercase and removing special characters.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text with only alphanumeric characters and spaces.
    """
    # Convert the text to lowercase
    text = text.lower()
    # Remove special characters, keeping only alphanumeric characters and spaces
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text


# Function to preprocess all chunks
def preprocess_chunks(chunks: List[str]) -> List[str]:
    """
    Apply preprocessing to all text chunks.

    Args:
        chunks (List[str]): A list of text chunks to preprocess.

    Returns:
        List[str]: A list of preprocessed text chunks.
    """
    # Apply the preprocess_text function to each chunk in the list
    return [preprocess_text(chunk) for chunk in chunks]

    
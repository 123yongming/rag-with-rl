import os
from openai import OpenAI
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
from config import client


# Function to construct a prompt with context
def construct_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Construct a prompt by combining the query with the retrieved context chunks.

    Args:
        query (str): The query text for which the prompt is being constructed.
        context_chunks (List[str]): A list of relevant context chunks to include in the prompt.

    Returns:
        str: The constructed prompt to be used as input for the LLM.
    """
    # Combine all context chunks into a single string, separated by newlines
    context = "\n".join(context_chunks)
    
    # Define the system message to guide the LLM's behavior
    system_message = (
        "You are a helpful assistant. Only use the provided context to answer the question. "
        "If the context doesn't contain the information needed, say 'I don't have enough information to answer this question.'"
    )
    
    # Construct the final prompt by combining the system message, context, and query
    prompt = f"System: {system_message}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    
    return prompt

# Function to generate a response using the OpenAI chat model
def generate_response(
    prompt: str,
    model: str = "deepseek-ai/DeepSeek-V3.2",
    max_tokens: int = 512,
    temperature: float = 1,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
    """
    Generate a response from the OpenAI chat model based on the constructed prompt.

    Args:
        prompt (str): The input prompt to provide to the chat model.
        model (str): The model to use for generating the response. Default is "google/gemma-2-2b-it".
        max_tokens (int): Maximum number of tokens in the response. Default is 512.
        temperature (float): Sampling temperature for response diversity. Default is 0.5.
        top_p (float): Probability mass for nucleus sampling. Default is 0.9.
        top_k (int): Number of highest probability tokens to consider. Default is 50.

    Returns:
        str: The generated response from the chat model.
    """
    # Use the OpenAI client to create a chat completion
    response = client.chat.completions.create(
        model=model,  # Specify the model to use for generating the response
        max_tokens=max_tokens,  # Maximum number of tokens in the response
        temperature=temperature,  # Sampling temperature for response diversity
        top_p=top_p,  # Probability mass for nucleus sampling
        extra_body={  # Additional parameters for the request
            "top_k": top_k  # Number of highest probability tokens to consider
        },
        messages=[  # List of messages to provide context for the chat model
            {
                "role": "user",  # Role of the message sender (user in this case)
                "content": [  # Content of the message
                    {
                        "type": "text",  # Type of content (text in this case)
                        "text": prompt  # The actual prompt text
                    }
                ]
            }
        ]
    )
    # Return the content of the first choice in the response
    return response.choices[0].message.content
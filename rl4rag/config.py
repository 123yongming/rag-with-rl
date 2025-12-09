import os
from openai import OpenAI
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union

# Specify the directory path containing the text files
directory_path = "data"

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1/",
    api_key='sk-xlsqognrjytfdkwloincmvsgrihqigmncxnfsccvonhutzuh'
)

# Initialize an in-memory vector store as a dictionary
# The keys will be unique identifiers (integers), and the values will be dictionaries containing embeddings and corresponding text chunks
vector_store: dict[int, dict[str, object]] = {}
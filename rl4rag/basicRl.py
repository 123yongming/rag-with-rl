'''
    TODO List
    价值函数存在较大问题，如何正确使用价值函数判断？（能否使用fassi计算？）
    使用grpo实现rl
'''


import os
from openai import OpenAI
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
from faissStoreUtils import *
from llmUtils import *
from pipelines import *

# Function to define the state representation for reinforcement learning
def define_state(
    query: str, 
    context_chunks: List[str], 
    rewritten_query: str = None, 
    previous_responses: List[str] = None, 
    previous_rewards: List[float] = None
) -> dict:
    """
    Define the state representation for the reinforcement learning agent.
    
    Args:
        query (str): The original user query.
        context_chunks (List[str]): Retrieved context chunks from the knowledge base.
        rewritten_query (str, optional): A reformulated version of the original query.
        previous_responses (List[str], optional): List of previously generated responses.
        previous_rewards (List[float], optional): List of rewards received for previous actions.
    
    Returns:
        dict: A dictionary representing the current state with all relevant information.
    """
    state = {
        "original_query": query,                                    # The initial query from the user
        "current_query": rewritten_query if rewritten_query else query,  # Current version of the query (may be rewritten)
        "context": context_chunks,                                 # Retrieved context chunks from the knowledge base
        "previous_responses": previous_responses if previous_responses else [],  # History of generated responses
        "previous_rewards": previous_rewards if previous_rewards else []         # History of received rewards
    }
    return state



# Function to define the action space for reinforcement learning
def define_action_space() -> List[str]:
    """
    Define the set of possible actions the reinforcement learning agent can take.
    
    Actions include:
    - rewrite_query: Reformulate the original query to improve retrieval
    - expand_context: Retrieve additional context chunks
    - filter_context: Remove irrelevant context chunks
    - generate_response: Generate a response based on current query and context
    
    Returns:
        List[str]: A list of available actions.
    """

    # Define the set of actions the agent can take
    actions = ["rewrite_query", "expand_context", "filter_context", "generate_response"]
    return actions


# Function to calculate the reward based on response quality
def calculate_reward(response: str, ground_truth: str) -> float:
    """
    Calculate a reward value by comparing the generated response to the ground truth.
    
    Uses cosine similarity between the embeddings of the response and ground truth
    to determine how close the response is to the expected answer.
    
    Args:
        response (str): The generated response from the RAG pipeline.
        ground_truth (str): The expected correct answer.
    
    Returns:
        float: A reward value between -1 and 1, where higher values indicate 
               greater similarity to the ground truth.
    """
    # Generate embeddings for both the response and ground truth
    response_embedding = generate_embeddings([response])[0]
    ground_truth_embedding = generate_embeddings([ground_truth])[0]
    
    # Calculate cosine similarity between the embeddings as the reward
    similarity = cosine_similarity(response_embedding, ground_truth_embedding)
    return similarity

# Function to rewrite the query for better document retrieval
def rewrite_query(
    query: str, 
    context_chunks: List[str], 
    model: str = "deepseek-ai/DeepSeek-V3.2", 
    max_tokens: int = 100, 
    temperature: float = 0.3
) -> str:
    """
    Use the LLM to rewrite the query for better document retrieval.

    Args:
        query (str): The original query text.
        context_chunks (List[str]): A list of context chunks retrieved so far.
        model (str): The model to use for generating the rewritten query. Default is "google/gemma-2-2b-it".
        max_tokens (int): Maximum number of tokens in the rewritten query. Default is 100.
        temperature (float): Sampling temperature for response diversity. Default is 0.3.

    Returns:
        str: The rewritten query optimized for document retrieval.
    """
    # Construct a prompt for the LLM to rewrite the query
    rewrite_prompt = f"""
    You are a query optimization assistant. Your task is to rewrite the given query to make it more effective 
    for retrieving relevant information. The query will be used for document retrieval.
    
    Original query: {query}
    
    Based on the context retrieved so far:
    {' '.join(context_chunks[:2]) if context_chunks else 'No context available yet'}
    
    Rewrite the query to be more specific and targeted to retrieve better information.
    Rewritten query:
    """
    
    # Use the LLM to generate a rewritten query
    response = client.chat.completions.create(
        model=model, # Specify the model to use for generating the response
        max_tokens=max_tokens, # Maximum number of tokens in the response
        temperature=temperature, # Sampling temperature for response diversity
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt
            }
        ]
    )
    
    # Extract and return the rewritten query from the response
    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query


# Function to expand the context by retrieving additional chunks
def expand_context(query: str, current_chunks: List[str], top_k: int = 3) -> List[str]:
    """
    Expand the context by retrieving additional chunks.

    Args:
        query (str): The query text for which additional context is needed.
        current_chunks (List[str]): The current list of context chunks.
        top_k (int): The number of additional chunks to retrieve. Default is 3.

    Returns:
        List[str]: The expanded list of context chunks including new unique chunks.
    """
    # Retrieve more chunks than currently available
    additional_chunks = retrieve_relevant_chunks_faiss(query, top_k=top_k + len(current_chunks))
    
    # Filter out chunks that are already in the current context
    new_chunks = []
    for chunk in additional_chunks:
        if chunk not in current_chunks:
            new_chunks.append(chunk)
    
    # Add new unique chunks to the current context, limited to top_k
    expanded_context = current_chunks + new_chunks[:top_k]
    return expanded_context


# Function to filter the context to keep only the most relevant chunks
def filter_context(query: str, context_chunks: List[str]) -> List[str]:
    """
    Filter the context to keep only the most relevant chunks.

    Args:
        query (str): The query text for which relevance is calculated.
        context_chunks (List[str]): The list of context chunks to filter.

    Returns:
        List[str]: A filtered list of the most relevant context chunks.
    """
    if not context_chunks:
        return []
        
    # Generate embeddings for the query and each chunk
    query_embedding = generate_embeddings([query])[0]
    chunk_embeddings = [generate_embeddings([chunk])[0] for chunk in context_chunks]
    
    # Calculate relevance scores for each chunk
    relevance_scores = []
    for chunk_embedding in chunk_embeddings:
        score = cosine_similarity(query_embedding, chunk_embedding)
        relevance_scores.append(score)
    
    # Sort chunks by relevance scores in descending order
    sorted_chunks = [x for _, x in sorted(zip(relevance_scores, context_chunks), reverse=True)]
    
    # Keep the top 5 most relevant chunks or fewer if less than 5 are available
    filtered_chunks = sorted_chunks[:min(5, len(sorted_chunks))]
    
    return filtered_chunks


# Function to define a policy network to select an action based on the state
def policy_network(
    state: dict, 
    action_space: List[str], 
    epsilon: float = 0.2
) -> str:
    """
    Define a policy network to select an action based on the current state using an epsilon-greedy strategy.

    Args:
        state (dict): The current state of the environment, including query, context, responses, and rewards.
        action_space (List[str]): The list of possible actions the agent can take.
        epsilon (float): The probability of choosing a random action for exploration. Default is 0.2.

    Returns:
        str: The selected action from the action space.
    """
    # Use epsilon-greedy strategy: random exploration vs. exploitation
    if np.random.random() < epsilon:
        # Exploration: randomly select an action from the action space
        action = np.random.choice(action_space)
    else:
        # Exploitation: select the best action based on the current state using a simple heuristic

        # If there are no previous responses, prioritize rewriting the query
        if len(state["previous_responses"]) == 0:
            action = "rewrite_query"
        # If there are previous responses but the rewards are low, try expanding the context
        elif state["previous_rewards"] and max(state["previous_rewards"]) < 0.7:
            action = "expand_context"
        # If the context has too many chunks, try filtering the context
        elif len(state["context"]) > 5:
            action = "filter_context"
        # Otherwise, generate a response
        else:
            action = "generate_response"
    
    return action


# Function to perform a single RL step
def rl_step(
    state: dict, 
    action_space: List[str], 
    ground_truth: str
) -> tuple[dict, str, float, str]:
    """
    Perform a single RL step: select an action, execute it, and calculate the reward.

    Args:
        state (dict): The current state of the environment, including query, context, responses, and rewards.
        action_space (List[str]): The list of possible actions the agent can take.
        ground_truth (str): The expected correct answer to calculate the reward.

    Returns:
        tuple: A tuple containing:
            - state (dict): The updated state after executing the action.
            - action (str): The action selected by the policy network.
            - reward (float): The reward received for the action.
            - response (str): The response generated (if applicable).
    """
    # Select an action using the policy network
    action: str = policy_network(state, action_space)
    response: str = None  # Initialize response as None
    reward: float = 0  # Initialize reward as 0

    # Execute the selected action
    if action == "rewrite_query":
        # Rewrite the query to improve retrieval
        rewritten_query: str = rewrite_query(state["original_query"], state["context"])
        state["current_query"] = rewritten_query  # Update the current query in the state
        # Retrieve new context based on the rewritten query
        new_context: List[str] = retrieve_relevant_chunks_faiss(rewritten_query)
        state["context"] = new_context  # Update the context in the state

    elif action == "expand_context":
        # Expand the context by retrieving additional chunks
        expanded_context: List[str] = expand_context(state["current_query"], state["context"])
        state["context"] = expanded_context  # Update the context in the state

    elif action == "filter_context":
        # Filter the context to keep only the most relevant chunks
        filtered_context: List[str] = filter_context(state["current_query"], state["context"])
        state["context"] = filtered_context  # Update the context in the state

    elif action == "generate_response":
        # Construct a prompt using the current query and context
        prompt: str = construct_prompt(state["current_query"], state["context"])
        # Generate a response using the LLM
        response: str = generate_response(prompt)
        # Calculate the reward based on the similarity between the response and the ground truth
        reward: float = calculate_reward(response, ground_truth)
        # Update the state with the new response and reward
        state["previous_responses"].append(response)
        state["previous_rewards"].append(reward)

    # Return the updated state, selected action, reward, and response
    return state, action, reward, response


# Function to initialize training parameters
def initialize_training_params() -> Dict[str, Union[float, int]]:
    """
    Initialize training parameters such as learning rate, number of episodes, and discount factor.

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing the initialized training parameters.
    """
    params = {
        "learning_rate": 0.01,  # Learning rate for policy updates
        "num_episodes": 100,   # Total number of training episodes
        "discount_factor": 0.99  # Discount factor for future rewards
    }
    return params


# Function to update policy based on reward
def update_policy(
    policy: Dict[str, Dict[str, Union[float, str]]], 
    state: Dict[str, object], 
    action: str, 
    reward: float, 
    learning_rate: float
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Update the policy based on the reward received.

    Args:
        policy (Dict[str, Dict[str, Union[float, str]]]): The current policy to be updated.
        state (Dict[str, object]): The current state of the environment.
        action (str): The action taken by the agent.
        reward (float): The reward received for the action.
        learning_rate (float): The learning rate for updating the policy.

    Returns:
        Dict[str, Dict[str, Union[float, str]]]: The updated policy.
    """
    # Example: Simple policy update (to be replaced with a proper RL algorithm)
    policy[state["query"]] = {
        "action": action,  # Store the action taken
        "reward": reward   # Store the reward received
    }
    return policy


# Function to track training progress
def track_progress(
    episode: int, 
    reward: float, 
    rewards_history: List[float]
) -> List[float]:
    """
    Track the training progress by storing rewards for each episode.

    Args:
        episode (int): The current episode number.
        reward (float): The reward received in the current episode.
        rewards_history (List[float]): A list to store the rewards for all episodes.

    Returns:
        List[float]: The updated rewards history.
    """
    # Append the current reward to the rewards history
    rewards_history.append(reward)
    
    # Print progress every 10 episodes
    print(f"Episode {episode}: Reward = {reward}")
    
    return rewards_history




# Function to implement the training loop
def training_loop(
    query_text: str, 
    ground_truth: str, 
    params: Optional[Dict[str, Union[float, int]]] = None
) -> Tuple[Dict[str, Dict[str, Union[float, str]]], List[float], List[List[str]], Optional[str]]:
    """
    Implement the training loop for RL-enhanced RAG.

    Args:
        query_text (str): The input query text for the RAG pipeline.
        ground_truth (str): The expected correct answer for the query.
        params (Optional[Dict[str, Union[float, int]]]): Training parameters such as learning rate, 
            number of episodes, and discount factor. If None, default parameters are initialized.

    Returns:
        Tuple: A tuple containing:
            - policy (Dict[str, Dict[str, Union[float, str]]]): The updated policy after training.
            - rewards_history (List[float]): A list of rewards received in each episode.
            - actions_history (List[List[str]]): A list of actions taken in each episode.
            - best_response (Optional[str]): The best response generated during training.
    """
    # Initialize training parameters if not provided
    if params is None:
        params = initialize_training_params()
    
    # Initialize variables to track progress
    rewards_history: List[float] = []  # List to store rewards for each episode
    actions_history: List[List[str]] = []  # List to store actions taken in each episode
    policy: Dict[str, Dict[str, Union[float, str]]] = {}  # Policy dictionary to store actions and rewards
    action_space: List[str] = define_action_space()  # Define the action space
    best_response: Optional[str] = None  # Variable to store the best response
    best_reward: float = -1  # Initialize the best reward to a very low value
    
    # Get initial performance from the simple RAG pipeline for comparison
    simple_response: str = basic_rag_pipeline(query_text)
    simple_reward: float = calculate_reward(simple_response, ground_truth)
    print(f"Simple RAG reward: {simple_reward:.4f}")

    # Start the training loop
    for episode in range(params["num_episodes"]):
        # Reset the environment with the same query
        context_chunks: List[str] = retrieve_relevant_chunks_faiss(query_text)
        state: Dict[str, object] = define_state(query_text, context_chunks)
        episode_reward: float = 0  # Initialize the reward for the current episode
        episode_actions: List[str] = []  # Initialize the list of actions for the current episode
        
        # Maximum number of steps per episode to prevent infinite loops
        for step in range(10):
            # Perform a single RL step
            state, action, reward, response = rl_step(state, action_space, ground_truth)
            episode_actions.append(action)  # Record the action taken
            
            # If a response is generated, end the episode
            if response:
                episode_reward = reward  # Update the episode reward
                
                # Track the best response and reward
                if reward > best_reward:
                    best_reward = reward
                    best_response = response
                
                break  # Exit the loop as the episode ends
        
        # Update rewards and actions history
        rewards_history.append(episode_reward)
        actions_history.append(episode_actions)
        
        # Track progress using the track_progress function
        track_progress(episode, episode_reward, rewards_history)
    
    # Compare the best RL-enhanced RAG reward with the simple RAG reward
    improvement: float = best_reward - simple_reward
    print(f"\nTraining completed:")
    print(f"Simple RAG reward: {simple_reward:.4f}")
    print(f"Best RL-enhanced RAG reward: {best_reward:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement * 100:.2f}%)")

    return policy, rewards_history, actions_history, best_response



# Function to compare Simple RAG vs RL-Enhanced RAG
def compare_rag_approaches(query_text: str, ground_truth: str) -> Tuple[str, str, float, float]:
    """
    Compare the outputs of simple RAG versus RL-enhanced RAG.

    Args:
        query_text (str): The input query text for the RAG pipeline.
        ground_truth (str): The expected correct answer for the query.

    Returns:
        Tuple[str, str, float, float]: A tuple containing:
            - simple_response (str): The response generated by the simple RAG pipeline.
            - best_rl_response (str): The best response generated by the RL-enhanced RAG pipeline.
            - simple_similarity (float): The similarity score of the simple RAG response to the ground truth.
            - rl_similarity (float): The similarity score of the RL-enhanced RAG response to the ground truth.
    """
    print("=" * 80)
    print(f"Query: {query_text}")
    print("=" * 80)
    
    # Step 1: Generate a response using the simple RAG pipeline
    # The basic RAG pipeline retrieves relevant chunks and generates a response without reinforcement learning.
    simple_response: str = basic_rag_pipeline(query_text)
    # Calculate the similarity score between the simple RAG response and the ground truth.
    simple_similarity: float = calculate_reward(simple_response, ground_truth)
    
    print("\nSimple RAG Output:")
    print("-" * 40)
    print(simple_response)
    print(f"Similarity to ground truth: {simple_similarity:.4f}")
    
    # Step 2: Train the RL-enhanced RAG model
    print("\nTraining RL-enhanced RAG model...")
    # Initialize training parameters (e.g., learning rate, number of episodes, discount factor).
    params: Dict[str, float | int] = initialize_training_params()
    # Set the number of episodes to a smaller value for demonstration purposes.
    params["num_episodes"] = 5
    
    # Run the training loop for the RL-enhanced RAG model.
    # This loop trains the model to optimize its responses using reinforcement learning.
    _, rewards_history, actions_history, best_rl_response = training_loop(
        query_text, ground_truth, params
    )
    
    # If no response was generated during training, generate one using the current query and context.
    if best_rl_response is None:
        # Retrieve relevant chunks for the query.
        context_chunks: List[str] = retrieve_relevant_chunks(query_text)
        # Construct a prompt using the query and retrieved context.
        prompt: str = construct_prompt(query_text, context_chunks)
        # Generate a response using the language model.
        best_rl_response: str = generate_response(prompt)
    
    # Calculate the similarity score between the RL-enhanced RAG response and the ground truth.
    rl_similarity: float = calculate_reward(best_rl_response, ground_truth)
    
    print("\nRL-enhanced RAG Output:")
    print("-" * 40)
    print(best_rl_response)
    print(f"Similarity to ground truth: {rl_similarity:.4f}")
    
    # Step 3: Evaluate and compare the results
    # Calculate the improvement in similarity score achieved by the RL-enhanced RAG model.
    improvement: float = rl_similarity - simple_similarity
    
    print("\nEvaluation Results:")
    print("-" * 40)
    print(f"Simple RAG similarity to ground truth: {simple_similarity:.4f}")
    print(f"RL-enhanced RAG similarity to ground truth: {rl_similarity:.4f}")
    print(f"Improvement: {improvement * 100:.2f}%")
    
    # Step 4: Plot the reward history (if there are enough episodes and matplotlib is available)
    if len(rewards_history) > 1:
        try:
            import matplotlib.pyplot as plt
            # Create a plot to visualize the reward history during RL training.
            plt.figure(figsize=(10, 6))
            plt.plot(rewards_history)
            plt.title('Reward History During RL Training')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.show()
        except ImportError:
            # If matplotlib is not available, print a message instead of plotting.
            print("Matplotlib not available for plotting rewards")
    
    # Return the results: responses and similarity scores for both approaches.
    return simple_response, best_rl_response, simple_similarity, rl_similarity
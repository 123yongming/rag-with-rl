'''
    TODO List
    整理目前的价值函数实现
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

# 导入高级奖励系统
try:
    from advanced_reward_system import (
        AdvancedRewardCalculator, 
        RewardConfig, 
        create_advanced_reward_calculator,
        REWARD_CONFIGS,
        integrate_with_basic_rl
    )
    ADVANCED_REWARD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import advanced reward system: {e}")
    ADVANCED_REWARD_AVAILABLE = False

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
def calculate_reward(
    response: str = None, 
    ground_truth: str = None, 
    env_state: dict = None, 
    action: str = None, 
    feedback_result: dict = None
) -> float:
    """
    Calculate a reward value using the advanced reward system.
    
    Args:
        response (str): The generated response from the RAG pipeline.
        ground_truth (str): The expected correct answer.
        env_state (dict): The current environment state containing query, context, etc.
        action (str): The action that was taken in the environment.
        feedback_result (dict): Additional feedback from the environment.
    
    Returns:
        float: A reward value in the range [-1, 1].
        
    Raises:
        ValueError: If advanced reward system is not available or required parameters are missing.
        RuntimeError: If advanced reward system calculation fails.
    """
    # Check if advanced reward system is available
    if not ADVANCED_REWARD_AVAILABLE:
        raise RuntimeError("Advanced reward system is not available. Please ensure it's properly configured.")
    
    # Validate required parameters for advanced reward calculation
    if env_state is None:
        raise ValueError("env_state is required for advanced reward calculation")
    if action is None:
        raise ValueError("action is required for advanced reward calculation")
    if response is None:
        raise ValueError("response is required for advanced reward calculation")
    if ground_truth is None:
        raise ValueError("ground_truth is required for advanced reward calculation")
    
    try:
        # Get the advanced reward calculator
        calculator = _get_advanced_calculator()
        if calculator is None:
            raise RuntimeError("Advanced reward calculator could not be initialized")
        
        # Prepare reward input data
        reward_input = {
            'query': env_state.get('query', ''),
            'context': env_state.get('context', []),
            'response': response,
            'ground_truth': ground_truth,
            'action': action,
            'retrieval_results': env_state.get('retrieval_results', []),
            'generation_results': env_state.get('generation_results', []),
            'feedback': feedback_result or {}
        }
        
        # Extract parameters for advanced system
        query = reward_input.get('query', '')
        context = reward_input.get('context', [])
        response_text = reward_input.get('response', '')
        ground_truth_text = reward_input.get('ground_truth', '')
        action_str = reward_input.get('action', '')
        
        # Calculate comprehensive reward using advanced system
        result = calculator.calculate_comprehensive_reward(
            query=query,
            retrieved_chunks=context,
            response=response_text,
            ground_truth=ground_truth_text,
            state=env_state,
            action=action_str,
            step_number=1,  # Default step number
            is_final_step=False  # Default to non-final step
        )
        
        # Extract the overall reward
        overall_reward = result.get('overall_reward', 0.0)
        
        # Convert to [-1, 1] range to maintain compatibility with existing system
        # Advanced system typically returns [0, 1], so we transform it
        normalized_reward = (overall_reward - 0.5) * 2.0
        
        return max(-1.0, min(1.0, normalized_reward))
        
    except Exception as e:
        # Log the error and raise a more descriptive exception
        error_msg = f"Advanced reward calculation failed: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

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
    reward: float = 0.0  # Initialize reward as 0.0 (浮点数)

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
        # Create env_state for reward calculation
        env_state = {
            'query': state.get("current_query", ""),
            'context': state.get("context", []),
            'retrieval_results': state.get("context", []),
            'previous_rewards': state.get("previous_rewards", [])
        }
        reward: float = calculate_reward(
            response=response,
            ground_truth=ground_truth,
            env_state=env_state,
            action=action
        )
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
    # Get context for simple RAG to create proper env_state
    simple_context = retrieve_relevant_chunks_faiss(query_text)
    simple_env_state = {
        'query': query_text,
        'context': simple_context,
        'retrieval_results': simple_context,
        'previous_rewards': []
    }
    simple_reward: float = calculate_reward(
        response=simple_response,
        ground_truth=ground_truth,
        env_state=simple_env_state,
        action='generate_response'
    )
    print(f"Simple RAG reward: {simple_reward:.4f}")

    # Start the training loop
    for episode in range(params["num_episodes"]):
        # Reset the environment with the same query
        context_chunks: List[str] = retrieve_relevant_chunks_faiss(query_text)
        state: Dict[str, object] = define_state(query_text, context_chunks)
        episode_reward: float = 0.0  # Initialize the reward for the current episode (浮点数)
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
    
    # Get context for simple RAG to create proper env_state
    simple_context = retrieve_relevant_chunks_faiss(query_text)
    simple_env_state = {
        'query': query_text,
        'context': simple_context,
        'retrieval_results': simple_context,
        'previous_rewards': []
    }
    
    # Calculate the similarity score between the simple RAG response and the ground truth.
    simple_similarity: float = calculate_reward(
        response=simple_response,
        ground_truth=ground_truth,
        env_state=simple_env_state,
        action='generate_response'
    )
    
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
        context_chunks: List[str] = retrieve_relevant_chunks_faiss(query_text)
        # Construct a prompt using the query and retrieved context.
        prompt: str = construct_prompt(query_text, context_chunks)
        # Generate a response using the language model.
        best_rl_response: str = generate_response(prompt)
    
    # Calculate the similarity score between the RL-enhanced RAG response and the ground truth.
    rl_context = retrieve_relevant_chunks_faiss(query_text)
    rl_env_state = {
        'query': query_text,
        'context': rl_context,
        'retrieval_results': rl_context,
        'previous_rewards': []
    }
    rl_similarity: float = calculate_reward(
        response=best_rl_response,
        ground_truth=ground_truth,
        env_state=rl_env_state,
        action='generate_response'
    )
    
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


# 初始化高级奖励计算器
_global_advanced_calculator = None

def _get_advanced_calculator():
    """获取全局高级奖励计算器实例"""
    global _global_advanced_calculator
    if _global_advanced_calculator is None and ADVANCED_REWARD_AVAILABLE:
        config = REWARD_CONFIGS.get('balanced', RewardConfig())
        _global_advanced_calculator = create_advanced_reward_calculator({
            'relevance_weight': 0.3,
            'coverage_weight': 0.2,
            'accuracy_weight': 0.25,
            'fluency_weight': 0.1,
            'completeness_weight': 0.15,
            'bias_correction': True
        })
    return _global_advanced_calculator
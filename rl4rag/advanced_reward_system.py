"""
Advanced Reward System for RL-RAG
================================

This module implements a comprehensive reward system for reinforcement learning 
enhanced retrieval-augmented generation, incorporating multi-stage rewards,
reward shaping, intrinsic motivation, and bias mitigation techniques.

Based on latest research from 2023-2024, including:
- CORAG (Chain-of-Retrieval Augmented Generation)
- Text2Reward Framework
- RAG 2.0 Technology Framework
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque
import math
from dataclasses import dataclass
from enum import Enum
try:
    from embeddingUtils import generate_embeddings
    from faissStoreUtils import cosine_similarity
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    # Fallback implementations
    def generate_embeddings(texts):
        """Fallback embedding generation using simple hash-based embeddings."""
        import hashlib
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            # Convert to fixed-size vector
            embedding = []
            for i in range(64):  # Fixed dimension
                embedding.append((hash_int >> i) & 1)
            embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)
    
    def cosine_similarity(vec1, vec2):
        """Fallback cosine similarity calculation."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)


class RewardStage(Enum):
    """Enumeration of different reward calculation stages."""
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    OVERALL = "overall"


@dataclass
class RewardConfig:
    """Configuration class for reward function parameters."""
    # Weight coefficients for different reward components
    relevance_weight: float = 0.3
    coverage_weight: float = 0.2
    diversity_weight: float = 0.1
    accuracy_weight: float = 0.25
    fluency_weight: float = 0.1
    completeness_weight: float = 0.15
    
    # Time discount factor for cumulative rewards
    discount_factor: float = 0.99
    
    # Reward shaping parameters
    shaping_coefficient: float = 0.1
    potential_threshold: float = 0.5
    
    # Intrinsic motivation parameters
    curiosity_weight: float = 0.05
    exploration_weight: float = 0.03
    
    # Bias mitigation parameters
    ensemble_size: int = 3
    bias_correction: bool = True


class MultiStageRewardCalculator:
    """
    Multi-stage reward calculator implementing retrieval and generation phase rewards.
    
    This class implements the core reward calculation logic based on the latest
    research findings from CORAG and RAG 2.0 frameworks.
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.state_visit_counts = defaultdict(int)
        self.query_history = []
        self.retrieval_performance_history = []
        
    def calculate_retrieval_reward(
        self, 
        query: str, 
        retrieved_chunks: List[str], 
        ground_truth_keywords: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate rewards for the retrieval stage.
        
        Args:
            query: The current query
            retrieved_chunks: List of retrieved document chunks
            ground_truth_keywords: Optional list of keywords for coverage calculation
            
        Returns:
            Dictionary containing different retrieval reward components
        """
        rewards = {}
        
        # 1. Relevance Reward (Semantic Similarity)
        rewards['relevance'] = self._calculate_relevance_reward(query, retrieved_chunks)
        
        # 2. Coverage Reward (Key Information Coverage)
        rewards['coverage'] = self._calculate_coverage_reward(
            query, retrieved_chunks, ground_truth_keywords
        )
        
        # 3. Diversity Reward (Semantic Diversity of Retrieved Chunks)
        rewards['diversity'] = self._calculate_diversity_reward(retrieved_chunks)
        
        # 4. Efficiency Reward (Computational Cost)
        rewards['efficiency'] = self._calculate_efficiency_reward(retrieved_chunks)
        
        # Combine retrieval rewards
        retrieval_reward = (
            self.config.relevance_weight * rewards['relevance'] +
            self.config.coverage_weight * rewards['coverage'] +
            self.config.diversity_weight * rewards['diversity'] +
            rewards['efficiency'] * 0.1  # Efficiency has lower weight
        )
        
        rewards['total_retrieval'] = retrieval_reward
        return rewards
    
    def calculate_generation_reward(
        self, 
        response: str, 
        ground_truth: str,
        retrieved_chunks: List[str],
        query: str
    ) -> Dict[str, float]:
        """
        Calculate rewards for the generation stage.
        
        Args:
            response: Generated response text
            ground_truth: Expected correct answer
            retrieved_chunks: Retrieved context chunks
            query: Original query
            
        Returns:
            Dictionary containing different generation reward components
        """
        rewards = {}
        
        # 1. Accuracy Reward (Fact Consistency)
        rewards['accuracy'] = self._calculate_accuracy_reward(
            response, ground_truth, retrieved_chunks
        )
        
        # 2. Fluency Reward (Language Model Perplexity)
        rewards['fluency'] = self._calculate_fluency_reward(response)
        
        # 3. Completeness Reward (Coverage of Query Requirements)
        rewards['completeness'] = self._calculate_completeness_reward(
            response, query, retrieved_chunks
        )
        
        # 4. Faithfulness Reward (Adherence to Retrieved Content)
        rewards['faithfulness'] = self._calculate_faithfulness_reward(
            response, retrieved_chunks
        )
        
        # Combine generation rewards
        generation_reward = (
            self.config.accuracy_weight * rewards['accuracy'] +
            self.config.fluency_weight * rewards['fluency'] +
            self.config.completeness_weight * rewards['completeness'] +
            rewards['faithfulness'] * 0.2  # Faithfulness has moderate weight
        )
        
        rewards['total_generation'] = generation_reward
        return rewards
    
    def _calculate_relevance_reward(self, query: str, chunks: List[str]) -> float:
        """Calculate semantic relevance between query and retrieved chunks."""
        try:
            # This would typically use embedding-based similarity
            # For now, using a simplified version
            
            if not chunks:
                return 0.0
                
            try:
                query_embedding = generate_embeddings([query])[0]
                chunk_embeddings = generate_embeddings(chunks)
            except Exception as e:
                print(f"Error generating embeddings for relevance reward: {e}")
                return 0.5  # Return neutral score on error
            
            similarities = []
            for chunk_emb in chunk_embeddings:
                try:
                    sim = cosine_similarity(query_embedding, chunk_emb)
                    similarities.append(sim)
                except Exception as e:
                    print(f"Error calculating cosine similarity: {e}")
                    similarities.append(0.0)
            
            # Return average similarity with diminishing returns for additional chunks
            if len(similarities) == 0:
                return 0.0
            
            # Weight more recent/top chunks higher
            weights = [0.4, 0.3, 0.2, 0.1][:len(similarities)]
            if len(weights) < len(similarities):
                weights.extend([0.05] * (len(similarities) - len(weights)))
            
            weighted_similarity = sum(s * w for s, w in zip(similarities, weights))
            return max(0.0, min(1.0, weighted_similarity))
        
        except Exception as e:
            print(f"Error in _calculate_relevance_reward: {e}")
            return 0.0
    
    def _calculate_coverage_reward(
        self, 
        query: str, 
        chunks: List[str], 
        keywords: List[str] = None
    ) -> float:
        """Calculate how well the retrieved chunks cover key information."""
        try:
            if not keywords:
                # Extract keywords from query (simplified approach)
                keywords = query.lower().split()
            
            if not chunks:
                return 0.0
                
            coverage_scores = []
            for chunk in chunks:
                try:
                    chunk_lower = chunk.lower()
                    keyword_matches = sum(1 for keyword in keywords if keyword in chunk_lower)
                    coverage_scores.append(keyword_matches / len(keywords))
                except Exception as e:
                    print(f"Error calculating coverage for chunk: {e}")
                    coverage_scores.append(0.0)
            
            # Return average coverage with bonus for high coverage
            if coverage_scores:
                avg_coverage = np.mean(coverage_scores)
                return min(1.0, avg_coverage * 1.2)  # Slight bonus for good coverage
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error in _calculate_coverage_reward: {e}")
            return 0.0
    
    def _calculate_diversity_reward(self, chunks: List[str]) -> float:
        """Calculate semantic diversity among retrieved chunks."""
        try:
            if len(chunks) <= 1:
                return 0.0
            
            try:
                chunk_embeddings = generate_embeddings(chunks)
            except Exception as e:
                print(f"Error generating embeddings for diversity reward: {e}")
                return 0.5  # Return neutral score on error
            
            # Calculate pairwise similarities
            similarities = []
            try:
                for i in range(len(chunk_embeddings)):
                    for j in range(i + 1, len(chunk_embeddings)):
                        try:
                            sim = cosine_similarity(chunk_embeddings[i], chunk_embeddings[j])
                            similarities.append(sim)
                        except Exception as e:
                            print(f"Error calculating similarity between chunks {i} and {j}: {e}")
                            similarities.append(0.0)
            except Exception as e:
                print(f"Error in pairwise similarity calculation: {e}")
                return 0.0
            
            # Higher diversity = lower average similarity
            if similarities:
                avg_similarity = np.mean(similarities)
                diversity_score = 1.0 - avg_similarity
                return max(0.0, diversity_score)
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error in _calculate_diversity_reward: {e}")
            return 0.0
    
    def _calculate_efficiency_reward(self, chunks: List[str]) -> float:
        """Calculate computational efficiency reward."""
        try:
            # Fewer chunks with good performance = higher efficiency
            optimal_chunk_count = 3  # Based on research findings
            
            if len(chunks) <= optimal_chunk_count:
                return 1.0 - (len(chunks) - 1) * 0.1
            else:
                # Penalty for excessive chunks
                penalty = (len(chunks) - optimal_chunk_count) * 0.1
                return max(0.0, 0.8 - penalty)
        except Exception as e:
            print(f"Error in _calculate_efficiency_reward: {e}")
            return 0.5  # Return neutral score on error
    
    def _calculate_accuracy_reward(
        self, 
        response: str, 
        ground_truth: str, 
        chunks: List[str]
    ) -> float:
        """Calculate factual accuracy reward."""
        try:
            try:
                # Use embedding similarity as proxy for semantic accuracy
                response_embedding = generate_embeddings([response])[0]
                truth_embedding = generate_embeddings([ground_truth])[0]
            except Exception as e:
                print(f"Error generating embeddings for accuracy reward: {e}")
                return 0.5  # Return neutral score on error
            
            try:
                semantic_similarity = cosine_similarity(response_embedding, truth_embedding)
            except Exception as e:
                print(f"Error calculating semantic similarity for accuracy: {e}")
                return 0.5
            
            # Also check consistency with retrieved content
            if chunks:
                try:
                    chunk_embeddings = generate_embeddings(chunks)
                    response_chunk_similarities = []
                    for chunk_emb in chunk_embeddings:
                        try:
                            similarity = cosine_similarity(response_embedding, chunk_emb)
                            response_chunk_similarities.append(similarity)
                        except Exception as e:
                            print(f"Error calculating chunk similarity for accuracy: {e}")
                            response_chunk_similarities.append(0.0)
                    
                    max_chunk_similarity = max(response_chunk_similarities) if response_chunk_similarities else 0.0
                    
                    # Reward consistency with retrieved content
                    consistency_bonus = max_chunk_similarity * 0.2
                    return min(1.0, semantic_similarity + consistency_bonus)
                except Exception as e:
                    print(f"Error in chunk similarity calculation for accuracy: {e}")
                    return semantic_similarity
            
            return semantic_similarity
            
        except Exception as e:
            print(f"Error in _calculate_accuracy_reward: {e}")
            return 0.5
    
    def _calculate_fluency_reward(self, response: str) -> float:
        """Calculate language fluency reward based on perplexity."""
        try:
            # This would typically use a language model to calculate perplexity
            # For now, using length-based heuristic as proxy
            words = response.split()
            
            # Optimal length range for fluency (based on research)
            optimal_length = 50
            if optimal_length > 0:
                length_score = 1.0 - abs(len(words) - optimal_length) / optimal_length
            else:
                length_score = 0.5
            
            # Bonus for proper sentence structure (simplified)
            sentence_count = response.count('.') + response.count('!') + response.count('?')
            if sentence_count > 0:
                try:
                    avg_sentence_length = len(words) / sentence_count
                    if avg_sentence_length > 0:
                        structure_score = 1.0 - abs(avg_sentence_length - 15) / 15
                        return max(0.0, (length_score + structure_score) / 2)
                    else:
                        return max(0.0, length_score)
                except Exception as e:
                    print(f"Error calculating structure score: {e}")
                    return max(0.0, length_score)
            
            return max(0.0, length_score)
            
        except Exception as e:
            print(f"Error in _calculate_fluency_reward: {e}")
            return 0.5  # Return neutral score on error
    
    def _calculate_completeness_reward(
        self, 
        response: str, 
        query: str, 
        chunks: List[str]
    ) -> float:
        """Calculate completeness of response relative to query requirements."""
        try:
            # Check if response addresses key aspects of the query
            query_keywords = set(query.lower().split())
            response_words = set(response.lower().split())
            
            # Handle division by zero if query is empty
            if len(query_keywords) == 0:
                return 0.5  # Neutral score for empty query
            
            keyword_coverage = len(query_keywords & response_words) / len(query_keywords)
            
            # Check for conclusion indicators
            conclusion_indicators = ['therefore', 'thus', 'in conclusion', 'finally', 'answer']
            has_conclusion = any(indicator in response.lower() for indicator in conclusion_indicators)
            
            completeness_score = keyword_coverage
            if has_conclusion:
                completeness_score *= 1.1  # Bonus for explicit conclusion
            
            return min(1.0, completeness_score)
            
        except Exception as e:
            print(f"Error in _calculate_completeness_reward: {e}")
            return 0.5  # Return neutral score on error
    
    def _calculate_faithfulness_reward(
        self, 
        response: str, 
        chunks: List[str]
    ) -> float:
        """Calculate faithfulness to retrieved content."""
        try:
            if not chunks:
                return 0.5  # Neutral score if no chunks available
            
            try:
                response_embedding = generate_embeddings([response])[0]
                chunk_embeddings = generate_embeddings(chunks)
            except Exception as e:
                print(f"Error generating embeddings for faithfulness reward: {e}")
                return 0.5  # Return neutral score on error
            
            # Check if response is consistent with retrieved content
            max_similarity = 0.0
            try:
                for chunk_emb in chunk_embeddings:
                    try:
                        similarity = cosine_similarity(response_embedding, chunk_emb)
                        max_similarity = max(max_similarity, similarity)
                    except Exception as e:
                        print(f"Error calculating similarity for faithfulness: {e}")
                        continue
            except Exception as e:
                print(f"Error in faithfulness similarity calculation: {e}")
                return 0.5
            
            return max_similarity
            
        except Exception as e:
            print(f"Error in _calculate_faithfulness_reward: {e}")
            return 0.5


class RewardShapingEngine:
    """
    Engine for implementing reward shaping techniques to handle sparse rewards.
    
    Implements potential-based reward shaping and intrinsic motivation mechanisms.
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.state_potentials = {}
        self.novelty_cache = {}
        self.state_visit_counts = {}  # Add missing state tracking
        
    def apply_reward_shaping(
        self, 
        state: Dict, 
        action: str, 
        reward: float, 
        next_state: Dict,
        step_number: int
    ) -> float:
        """
        Apply reward shaping to enhance learning signal.
        
        Args:
            state: Current state
            action: Action taken
            reward: Original reward
            next_state: Next state
            step_number: Current step in episode
            
        Returns:
            Shaped reward
        """
        try:
            # Track state visits for exploration rewards
            self._track_state_visits(next_state)
            
            # Calculate potential function value
            potential = self._calculate_potential(state, action, next_state)
            
            # Apply potential-based shaping
            shaped_reward = reward + self.config.shaping_coefficient * potential
            
            # Add intrinsic motivation rewards
            intrinsic_reward = self._calculate_intrinsic_reward(state, action, step_number)
            
            total_reward = shaped_reward + intrinsic_reward
            
            return max(0.0, min(1.0, total_reward))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"Error in apply_reward_shaping: {e}")
            return max(0.0, min(1.0, reward))  # Return clamped original reward on error
    
    def _track_state_visits(self, state: Dict) -> None:
        """Track state visits for exploration rewards."""
        try:
            state_key = self._hash_state(state)
            if state_key in self.state_visit_counts:
                self.state_visit_counts[state_key] += 1
            else:
                self.state_visit_counts[state_key] = 1
        except Exception as e:
            print(f"Error in _track_state_visits: {e}")
            # Don't raise exception to avoid breaking main flow
    
    def _calculate_potential(self, state: Dict, action: str, next_state: Dict) -> float:
        """Calculate potential function value for reward shaping."""
        try:
            # Potential based on improvement in state quality
            current_quality = self._assess_state_quality(state)
            next_quality = self._assess_state_quality(next_state)
            
            potential = next_quality - current_quality
            
            # Add action-specific potential
            action_potential = self._get_action_potential(action)
            potential += action_potential
            
            return potential
            
        except Exception as e:
            print(f"Error in _calculate_potential: {e}")
            return 0.0  # Return neutral potential on error
    
    def _calculate_intrinsic_reward(
        self, 
        state: Dict, 
        action: str, 
        step_number: int
    ) -> float:
        """Calculate intrinsic motivation reward."""
        try:
            intrinsic_reward = 0.0
            
            # Curiosity-driven reward (prediction error)
            curiosity_reward = self._calculate_curiosity_reward(state, action)
            intrinsic_reward += curiosity_reward * self.config.curiosity_weight
            
            # Exploration reward (novel state visit)
            exploration_reward = self._calculate_exploration_reward(state)
            intrinsic_reward += exploration_reward * self.config.exploration_weight
            
            return intrinsic_reward
            
        except Exception as e:
            print(f"Error in _calculate_intrinsic_reward: {e}")
            return 0.0  # Return neutral reward on error
    
    def _calculate_curiosity_reward(self, state: Dict, action: str) -> float:
        """Calculate curiosity-driven reward based on prediction error."""
        try:
            state_key = self._hash_state(state)
            
            # Simple prediction error based on state transition frequency
            if state_key in self.state_potentials:
                # High prediction error for novel transitions
                error = 1.0 / (1.0 + self.state_potentials[state_key])
                # Update potential with slight decay
                self.state_potentials[state_key] += 1
            else:
                error = 1.0  # Maximum curiosity for completely new state
                self.state_potentials[state_key] = 1
            
            return error
            
        except Exception as e:
            print(f"Error in _calculate_curiosity_reward: {e}")
            return 0.5  # Return neutral curiosity on error
    
    def _calculate_exploration_reward(self, state: Dict) -> float:
        """Calculate exploration reward based on state novelty."""
        try:
            state_key = self._hash_state(state)
            
            # Novel state gets higher reward
            if state_key not in self.state_visit_counts:
                return 1.0
            else:
                # Diminishing reward for frequently visited states
                visits = self.state_visit_counts[state_key]
                return 1.0 / math.sqrt(visits) if visits > 0 else 1.0
            
        except Exception as e:
            print(f"Error in _calculate_exploration_reward: {e}")
            return 0.5  # Return neutral exploration on error
    
    def _assess_state_quality(self, state: Dict) -> float:
        """Assess the quality of a state."""
        # Simple heuristic based on context quality
        if 'context' in state and state['context']:
            context_quality = min(len(state['context']) / 5.0, 1.0)  # Optimal context size = 5
        else:
            context_quality = 0.0
        
        # Query reformulation quality (if available)
        if 'current_query' in state and 'original_query' in state:
            if state['current_query'] != state['original_query']:
                query_quality = 0.8  # Bonus for query reformulation
            else:
                query_quality = 0.5
        else:
            query_quality = 0.5
        
        # Reward history quality
        if 'previous_rewards' in state and state['previous_rewards']:
            avg_reward = np.mean(state['previous_rewards'])
            reward_quality = (avg_reward + 1.0) / 2.0  # Normalize from [-1,1] to [0,1]
        else:
            reward_quality = 0.5
        
        return (context_quality + query_quality + reward_quality) / 3.0
    
    def _get_action_potential(self, action: str) -> float:
        """Get potential value for specific actions."""
        action_potentials = {
            'rewrite_query': 0.1,
            'expand_context': 0.05,
            'filter_context': 0.08,
            'generate_response': 0.15
        }
        return action_potentials.get(action, 0.0)
    
    def _hash_state(self, state: Dict) -> str:
        """Create a hashable representation of state."""
        # Create simplified state representation for hashing
        state_repr = {
            'query': state.get('current_query', ''),
            'context_size': len(state.get('context', [])),
            'num_responses': len(state.get('previous_responses', []))
        }
        return json.dumps(state_repr, sort_keys=True)


class BiasMitigationManager:
    """
    Manager for implementing bias mitigation techniques in reward calculation.
    
    Implements ensemble methods and bias correction mechanisms.
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.evaluator_weights = {}
        self.bias_patterns = defaultdict(list)
        self.historical_performance = defaultdict(list)  # Track performance over time
        self.bias_correction_factors = {}  # Dynamic correction factors
        
    def mitigate_bias(
        self, 
        rewards: Dict[str, float], 
        context: Dict,
        performance_feedback: float = None
    ) -> Dict[str, float]:
        """
        Apply bias mitigation techniques to reward calculations.
        
        Args:
            rewards: Original reward dictionary
            context: Context information for bias analysis
            performance_feedback: Optional performance feedback for adaptation
            
        Returns:
            Bias-corrected reward dictionary
        """
        try:
            corrected_rewards = rewards.copy()
            
            if self.config.bias_correction:
                # Apply bias correction
                corrected_rewards = self._apply_bias_correction(corrected_rewards, context)
            
            # Apply ensemble averaging if multiple evaluators
            if len(self.evaluator_weights) > 1:
                corrected_rewards = self._apply_ensemble_averaging(corrected_rewards)
            
            # Update bias correction factors if performance feedback is provided
            if performance_feedback is not None:
                bias_adjustments = self._detect_bias_patterns(context)
                self._update_bias_correction_factors(bias_adjustments, performance_feedback)
            
            return corrected_rewards
            
        except Exception as e:
            print(f"Error in mitigate_bias: {e}")
            return rewards  # Return original rewards on error
    
    def _apply_bias_correction(
        self, 
        rewards: Dict[str, float], 
        context: Dict
    ) -> Dict[str, float]:
        """Apply bias correction based on detected patterns."""
        try:
            # Detect potential biases in the current evaluation
            bias_adjustments = self._detect_bias_patterns(context)
            
            corrected_rewards = {}
            for reward_type, value in rewards.items():
                # Apply correction factor
                base_correction = bias_adjustments.get(reward_type, 1.0)
                adaptive_correction = self._get_adaptive_bias_correction(reward_type)
                correction_factor = base_correction * adaptive_correction
                corrected_value = value * correction_factor
                
                # Clamp to valid range
                corrected_rewards[reward_type] = max(0.0, min(1.0, corrected_value))
            
            return corrected_rewards
            
        except Exception as e:
            print(f"Error in _apply_bias_correction: {e}")
            return rewards  # Return original rewards on error
    
    def _detect_bias_patterns(self, context: Dict) -> Dict[str, float]:
        """Detect potential biases in the evaluation context."""
        adjustments = {}
        
        # Length bias correction
        if 'response_length' in context:
            length = context['response_length']
            if length > 200:  # Very long responses might be over-rewarded
                adjustments['fluency'] = 0.95
                adjustments['completeness'] = 0.98
            elif length < 10:  # Very short responses might be under-rewarded
                adjustments['accuracy'] = 1.05
                adjustments['completeness'] = 0.9
        
        # Domain bias correction
        if 'query_domain' in context:
            domain = context['query_domain']
            if domain == 'technical':
                adjustments['accuracy'] = 1.1  # Technical queries need higher accuracy
                adjustments['fluency'] = 0.95  # But might sacrifice some fluency
            elif domain == 'creative':
                adjustments['fluency'] = 1.1   # Creative queries need higher fluency
                adjustments['accuracy'] = 0.95  # Might sacrifice some accuracy
            elif domain == 'general':
                adjustments['relevance'] = 1.05  # General queries need balanced relevance
        
        # Response quality bias detection
        if 'retrieval_size' in context:
            retrieval_size = context['retrieval_size']
            if retrieval_size == 0:
                adjustments['relevance'] = 0.8  # Penalty for no retrieval
                adjustments['coverage'] = 0.5   # Heavy penalty for coverage
            elif retrieval_size > 10:
                adjustments['diversity'] = 1.1  # Bonus for diverse retrieval
                adjustments['efficiency'] = 0.9  # But might hurt efficiency
        
        # Context-dependent bias correction
        if 'previous_rewards' in context:
            prev_rewards = context['previous_rewards']
            if prev_rewards:
                avg_prev = np.mean(prev_rewards)
                if avg_prev > 0.8:  # High performance context
                    adjustments['reward_scale'] = 0.95  # Slight downward adjustment
                elif avg_prev < 0.2:  # Low performance context
                    adjustments['reward_scale'] = 1.05  # Slight upward adjustment
        
        return adjustments
    
    def _update_bias_correction_factors(self, bias_adjustments: Dict[str, float], performance_feedback: float) -> None:
        """Update bias correction factors based on performance feedback."""
        try:
            # Adjust correction factors based on performance
            for reward_type, adjustment in bias_adjustments.items():
                if reward_type in self.bias_correction_factors:
                    # Smooth adaptation: blend old and new factors
                    alpha = 0.1  # Learning rate for bias adaptation
                    old_factor = self.bias_correction_factors[reward_type]
                    new_factor = (1 - alpha) * old_factor + alpha * adjustment
                    self.bias_correction_factors[reward_type] = new_factor
                else:
                    self.bias_correction_factors[reward_type] = adjustment
            
            # Track performance for historical analysis
            performance_key = "overall_performance"
            self.historical_performance[performance_key].append(performance_feedback)
            
            # Keep only recent performance history (last 100 evaluations)
            if len(self.historical_performance[performance_key]) > 100:
                self.historical_performance[performance_key] = self.historical_performance[performance_key][-100:]
                
        except Exception as e:
            print(f"Error updating bias correction factors: {e}")
    
    def _get_adaptive_bias_correction(self, reward_type: str) -> float:
        """Get adaptive bias correction factor for a specific reward type."""
        try:
            # Start with default factor
            base_factor = 1.0
            
            # Apply historical adaptation if available
            if reward_type in self.bias_correction_factors:
                adaptive_factor = self.bias_correction_factors[reward_type]
            else:
                adaptive_factor = base_factor
            
            # Apply performance-based adjustment
            if reward_type in self.historical_performance:
                recent_performance = self.historical_performance[reward_type][-10:]  # Last 10 evaluations
                if recent_performance:
                    avg_performance = np.mean(recent_performance)
                    # Adjust factor based on performance trends
                    if avg_performance > 0.8:
                        adaptive_factor *= 0.98  # Slight reduction for consistently high performance
                    elif avg_performance < 0.3:
                        adaptive_factor *= 1.02  # Slight increase for consistently low performance
            
            return adaptive_factor
            
        except Exception as e:
            print(f"Error getting adaptive bias correction: {e}")
            return 1.0  # Return neutral factor on error
    
    def _apply_ensemble_averaging(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Apply ensemble averaging across multiple evaluators."""
        # This is a simplified version - in practice, you'd have multiple reward models
        averaged_rewards = {}
        
        for reward_type in rewards:
            # Weighted average across evaluators (simplified)
            # In practice, this would use actual predictions from multiple models
            weights = list(self.evaluator_weights.values())
            if weights:
                avg_weight = np.mean(weights)
                averaged_rewards[reward_type] = rewards[reward_type] * avg_weight
            else:
                averaged_rewards[reward_type] = rewards[reward_type]
        
        return averaged_rewards


class AdvancedRewardCalculator:
    """
    Main advanced reward calculator integrating all components.
    
    This class provides the main interface for calculating comprehensive rewards
    in RL-RAG systems.
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.multi_stage_calc = MultiStageRewardCalculator(self.config)
        self.shaping_engine = RewardShapingEngine(self.config)
        self.bias_manager = BiasMitigationManager(self.config)
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_steps = 0
        self.discount_factor = self.config.discount_factor
        
    def calculate_comprehensive_reward(
        self,
        query: str,
        retrieved_chunks: List[str],
        response: str,
        ground_truth: str,
        state: Dict,
        action: str,
        step_number: int,
        is_final_step: bool = False,
        ground_truth_keywords: List[str] = None
    ) -> Dict[str, Union[float, Dict]]:
        """
        Calculate comprehensive reward incorporating all components.
        
        Args:
            query: Current query
            retrieved_chunks: Retrieved document chunks
            response: Generated response
            ground_truth: Expected answer
            state: Current state
            action: Action taken
            step_number: Current step in episode
            is_final_step: Whether this is the final step
            ground_truth_keywords: Optional keywords for coverage calculation
            
        Returns:
            Comprehensive reward information
        """
        # Calculate stage-specific rewards
        retrieval_rewards = self.multi_stage_calc.calculate_retrieval_reward(
            query, retrieved_chunks, ground_truth_keywords
        )
        
        generation_rewards = self.multi_stage_calc.calculate_generation_reward(
            response, ground_truth, retrieved_chunks, query
        )
        
        # Calculate overall reward
        overall_reward = self._calculate_overall_reward(retrieval_rewards, generation_rewards)
        
        # Apply reward shaping
        shaped_reward = self.shaping_engine.apply_reward_shaping(
            state, action, overall_reward, state, step_number
        )
        
        # Apply bias mitigation
        context = {
            'response_length': len(response.split()),
            'query_domain': self._classify_query_domain(query),
            'retrieval_size': len(retrieved_chunks)
        }
        
        # Flatten rewards for bias mitigation
        flattened_rewards = {}
        for key, value in retrieval_rewards.items():
            flattened_rewards[f'retrieval_{key}'] = value
        for key, value in generation_rewards.items():
            flattened_rewards[f'generation_{key}'] = value
        # Also include overall reward
        flattened_rewards['overall'] = overall_reward
        
        corrected_rewards = self.bias_manager.mitigate_bias(
            flattened_rewards,
            context
        )
        
        # Calculate discounted cumulative reward
        if is_final_step:
            cumulative_reward = self._calculate_cumulative_reward()
        else:
            cumulative_reward = None
        
        # Compile comprehensive result
        reward_info = {
            'overall_reward': overall_reward,
            'shaped_reward': shaped_reward,
            'retrieval_rewards': retrieval_rewards,
            'generation_rewards': generation_rewards,
            'bias_corrected_rewards': corrected_rewards,
            'cumulative_reward': cumulative_reward,
            'step_number': step_number,
            'is_final_step': is_final_step
        }
        
        # Update episode tracking
        self.episode_rewards.append(shaped_reward)
        self.episode_steps += 1
        
        return reward_info
    
    def _calculate_overall_reward(
        self, 
        retrieval_rewards: Dict[str, float], 
        generation_rewards: Dict[str, float]
    ) -> float:
        """Calculate weighted overall reward from retrieval and generation components."""
        retrieval_total = retrieval_rewards.get('total_retrieval', 0.0)
        generation_total = generation_rewards.get('total_generation', 0.0)
        
        # Weights based on stage importance (generation slightly more important)
        retrieval_weight = 0.4
        generation_weight = 0.6
        
        overall = retrieval_weight * retrieval_total + generation_weight * generation_total
        return max(0.0, min(1.0, overall))
    
    def _calculate_cumulative_reward(self) -> float:
        """Calculate discounted cumulative reward for the episode."""
        if not self.episode_rewards:
            return 0.0
        
        cumulative = 0.0
        for i, reward in enumerate(self.episode_rewards):
            discounted_reward = reward * (self.discount_factor ** i)
            cumulative += discounted_reward
        
        return cumulative
    
    def _classify_query_domain(self, query: str) -> str:
        """Classify the domain of the query for bias mitigation."""
        query_lower = query.lower()
        
        # Simple domain classification based on keywords
        technical_keywords = ['code', 'algorithm', 'technical', 'implementation', 'function']
        creative_keywords = ['story', 'creative', 'write', 'imagine', 'design']
        
        if any(keyword in query_lower for keyword in technical_keywords):
            return 'technical'
        elif any(keyword in query_lower for keyword in creative_keywords):
            return 'creative'
        else:
            return 'general'
    
    def reset_episode(self):
        """Reset episode tracking for new episode."""
        self.episode_rewards = []
        self.episode_steps = 0
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about reward performance."""
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'total_episodes': len(self.episode_rewards),
            'cumulative_discounted_reward': self._calculate_cumulative_reward()
        }


# Example usage and integration functions
def create_advanced_reward_calculator(config_dict: Dict = None) -> AdvancedRewardCalculator:
    """
    Factory function to create an advanced reward calculator.
    
    Args:
        config_dict: Optional configuration dictionary
        
    Returns:
        Configured AdvancedRewardCalculator instance
    """
    if config_dict:
        config = RewardConfig(**config_dict)
    else:
        config = RewardConfig()
    
    return AdvancedRewardCalculator(config)


def integrate_with_basic_rl(
    basic_calculate_reward_func,
    advanced_calc: AdvancedRewardCalculator
):
    """
    Integration function to use advanced reward system with existing RL code.
    
    Args:
        basic_calculate_reward_func: Original calculate_reward function
        advanced_calc: Advanced reward calculator
        
    Returns:
        Enhanced reward calculation function
    """
    def enhanced_calculate_reward(
        response: str, 
        ground_truth: str,
        query: str = "",
        retrieved_chunks: List[str] = None,
        state: Dict = None,
        action: str = "",
        step_number: int = 0
    ):
        """Enhanced reward calculation using advanced system."""
        if retrieved_chunks is None:
            retrieved_chunks = []
        if state is None:
            state = {}
        if query == "":
            query = "What is the answer?"
        
        # Calculate comprehensive reward
        reward_info = advanced_calc.calculate_comprehensive_reward(
            query=query,
            retrieved_chunks=retrieved_chunks,
            response=response,
            ground_truth=ground_truth,
            state=state,
            action=action,
            step_number=step_number,
            is_final_step=True
        )
        
        # Return the shaped reward as the main reward value
        return reward_info['shaped_reward']
    
    return enhanced_calculate_reward


# Configuration presets for different use cases
REWARD_CONFIGS = {
    'balanced': RewardConfig(),
    'retrieval_focused': RewardConfig(
        relevance_weight=0.4,
        coverage_weight=0.3,
        accuracy_weight=0.2,
        fluency_weight=0.05,
        completeness_weight=0.05
    ),
    'generation_focused': RewardConfig(
        relevance_weight=0.2,
        coverage_weight=0.1,
        accuracy_weight=0.35,
        fluency_weight=0.2,
        completeness_weight=0.25
    ),
    'research_optimized': RewardConfig(
        relevance_weight=0.25,
        coverage_weight=0.2,
        diversity_weight=0.15,
        accuracy_weight=0.25,
        fluency_weight=0.1,
        completeness_weight=0.15,
        curiosity_weight=0.1,
        exploration_weight=0.05,
        bias_correction=True,
        ensemble_size=5
    )
}


if __name__ == "__main__":
    # Example usage
    config = REWARD_CONFIGS['research_optimized']
    reward_calc = AdvancedRewardCalculator(config)
    
    # Example query and context
    example_query = "What are the benefits of renewable energy?"
    example_chunks = [
        "Renewable energy sources like solar and wind provide clean electricity.",
        "Solar panels can reduce electricity bills significantly over time.",
        "Wind energy is one of the fastest-growing renewable energy sources."
    ]
    example_response = "Renewable energy offers multiple benefits including clean electricity generation, reduced electricity bills, and rapid growth potential."
    example_ground_truth = "Renewable energy provides clean power, cost savings, and represents a rapidly expanding sector."
    
    # Calculate reward
    reward_info = reward_calc.calculate_comprehensive_reward(
        query=example_query,
        retrieved_chunks=example_chunks,
        response=example_response,
        ground_truth=example_ground_truth,
        state={'context': example_chunks, 'previous_rewards': []},
        action='generate_response',
        step_number=1,
        is_final_step=True
    )
    
    print("Advanced Reward Calculation Results:")
    print(f"Overall Reward: {reward_info['overall_reward']:.4f}")
    print(f"Shaped Reward: {reward_info['shaped_reward']:.4f}")
    print(f"Retrieval Total: {reward_info['retrieval_rewards']['total_retrieval']:.4f}")
    print(f"Generation Total: {reward_info['generation_rewards']['total_generation']:.4f}")
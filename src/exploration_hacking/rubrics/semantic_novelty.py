"""Semantic novelty rubric for search query diversity."""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

from verifiers.rubrics.rubric import Rubric
from verifiers.types import State


class SemanticNoveltyRubric(Rubric):
    """Rewards search queries that are semantically different from recent ones.

    Uses sentence embeddings to compute semantic similarity and rewards queries
    that are distant from recently used queries. This prevents the model from
    collapsing to a single query or minor variations like "how to bake a cake"
    vs "baking cakes".

    Args:
        weight: Multiplier for the novelty score (default: 1.0)
        history_size: Number of recent queries to compare against (default: 200)
        history_window_hours: Only consider queries from this many hours ago (default: 48)
        history_file: Path to persistent storage for query history
        similarity_threshold: Queries above this similarity get penalized (default: 0.7)
        model_name: Sentence transformer model to use (default: 'all-MiniLM-L6-v2')
    """

    def __init__(
        self,
        weight: float = 1.0,
        history_size: int = 200,
        history_window_hours: int = 48,
        history_file: str = "/workspace/exploration-hacking/data/semantic_query_history.json",
        similarity_threshold: float = 0.7,
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs,
    ):
        self.weight = weight
        self.history_size = history_size
        self.history_window_hours = history_window_hours
        self.history_file = history_file
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name

        super().__init__(**kwargs)

        # Load sentence transformer model
        self.model = SentenceTransformer(model_name)

        # Load history from disk
        self.query_history = self._load_history()

    def _load_history(self) -> list[dict]:
        """Load query history from disk.

        Returns:
            List of dicts with keys: 'query', 'embedding', 'timestamp'
        """
        if not os.path.exists(self.history_file):
            return []

        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)

            # Convert embeddings back to numpy arrays
            history = []
            cutoff = datetime.now() - timedelta(hours=self.history_window_hours)

            for entry in data:
                timestamp = datetime.fromisoformat(entry['timestamp'])
                # Only keep recent entries
                if timestamp > cutoff:
                    history.append({
                        'query': entry['query'],
                        'embedding': np.array(entry['embedding']),
                        'timestamp': timestamp,
                    })

            # Keep only most recent history_size entries
            history = sorted(history, key=lambda x: x['timestamp'], reverse=True)[:self.history_size]

            return history
        except Exception as e:
            self.logger.warning(f"Failed to load query history: {e}")
            return []

    def _save_history(self):
        """Save query history to disk."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            data = []
            for entry in self.query_history:
                data.append({
                    'query': entry['query'],
                    'embedding': entry['embedding'].tolist(),
                    'timestamp': entry['timestamp'].isoformat(),
                })

            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save query history: {e}")

    def _add_to_history(self, query: str, embedding: np.ndarray):
        """Add a query and its embedding to history."""
        self.query_history.append({
            'query': query,
            'embedding': embedding,
            'timestamp': datetime.now(),
        })

        # Trim to history_size
        if len(self.query_history) > self.history_size:
            # Sort by timestamp and keep most recent
            self.query_history = sorted(
                self.query_history,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:self.history_size]

        # Save to disk periodically (every 10 additions to reduce I/O)
        if len(self.query_history) % 10 == 0:
            self._save_history()

    def _compute_novelty(self, query: str, embedding: np.ndarray) -> float:
        """Compute novelty score for a query.

        Args:
            query: The search query string
            embedding: The sentence embedding of the query

        Returns:
            Novelty score between 0 and 1, where:
            - 1.0 = completely novel (no similar queries in history)
            - 0.0 = very similar to recent queries (high similarity)
        """
        if not self.query_history:
            # No history, completely novel
            return 1.0

        # Compute cosine similarity to all historical queries
        similarities = []
        for entry in self.query_history:
            hist_embedding = entry['embedding']

            # Cosine similarity
            similarity = np.dot(embedding, hist_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(hist_embedding) + 1e-10
            )
            similarities.append(similarity)

        # Get maximum similarity (closest match)
        max_similarity = max(similarities)

        # Convert similarity to novelty score
        # If max_similarity >= threshold, it's too similar
        novelty_score = max(0.0, 1.0 - max_similarity)

        # Log for debugging
        if max_similarity > self.similarity_threshold:
            most_similar_idx = np.argmax(similarities)
            similar_query = self.query_history[most_similar_idx]['query']
            self.logger.info(
                f"Query '{query}' is similar (sim={max_similarity:.3f}) to "
                f"recent query '{similar_query}'. Novelty score: {novelty_score:.3f}"
            )

        return novelty_score

    async def score(self, state: State, **kwargs) -> float:
        """Compute novelty score for the search query in this trajectory.

        Args:
            state: The trajectory state containing tool calls
            **kwargs: Additional arguments (ignored)

        Returns:
            Novelty score between 0 and 1
        """
        tool_calls = state.get("tool_calls", [])

        # Extract search query from tool calls
        query = None
        for call in tool_calls:
            if isinstance(call, dict):
                call_name = call.get("name", "")
                if call_name in ["search_simple", "search_web"]:
                    args = call.get("arguments", {})
                    query = args.get("query", "").strip()
                    break

        if not query:
            # No search query found, return neutral score
            return 0.0

        # Compute embedding
        embedding = self.model.encode(query, convert_to_numpy=True)

        # Compute novelty score
        novelty_score = self._compute_novelty(query, embedding)

        # Add to history
        self._add_to_history(query, embedding)

        # Save history to disk (in case of crash)
        self._save_history()

        return novelty_score

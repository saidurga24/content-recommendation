"""
Collaborative Filtering Recommender
Implements User-based CF, Item-based CF, and Matrix Factorization (SVD)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib

from .config import CF_PARAMS, SVD_PARAMS, N_RECOMMENDATIONS, MODEL_DIR


class CollaborativeFiltering:
    """
    Collaborative Filtering Recommender System.

    Implements three approaches:
    1. User-based CF: Find similar users and recommend what they liked
    2. Item-based CF: Find similar items to what user has interacted with
    3. Matrix Factorization (SVD): Decompose user-item matrix for predictions
    """

    def __init__(self,
                 method: str = 'user_based',
                 n_neighbors: int = CF_PARAMS['n_neighbors'],
                 n_factors: int = SVD_PARAMS['n_factors']):
        """
        Initialize the collaborative filtering recommender.

        Args:
            method: 'user_based', 'item_based', or 'svd'
            n_neighbors: Number of similar users/items to consider
            n_factors: Number of latent factors for SVD
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.n_factors = n_factors

        # Model components
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None

        # ID mappings
        self.user_id_map = None
        self.article_id_map = None
        self.reverse_user_map = None
        self.reverse_article_map = None

        self.is_fitted = False

    def fit(self,
            user_item_matrix: csr_matrix,
            user_id_map: Dict,
            article_id_map: Dict) -> 'CollaborativeFiltering':
        """
        Fit the collaborative filtering model.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            user_id_map: Mapping from user IDs to matrix indices
            article_id_map: Mapping from article IDs to matrix indices

        Returns:
            self for method chaining
        """
        self.user_item_matrix = user_item_matrix
        self.user_id_map = user_id_map
        self.article_id_map = article_id_map
        self.reverse_user_map = {v: k for k, v in user_id_map.items()}
        self.reverse_article_map = {v: k for k, v in article_id_map.items()}

        if self.method == 'user_based':
            self._fit_user_based()
        elif self.method == 'item_based':
            self._fit_item_based()
        elif self.method == 'svd':
            self._fit_svd()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted = True
        return self

    def _fit_user_based(self):
        """Compute user-user similarity matrix."""
        print("Computing user-user similarity matrix...")
        # Convert to dense for similarity computation (can be memory intensive)
        # For large datasets, use approximate methods
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        print(f"User similarity matrix shape: {self.user_similarity.shape}")

    def _fit_item_based(self):
        """Compute item-item similarity matrix."""
        print("Computing item-item similarity matrix...")
        # Transpose for item-item similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        print(f"Item similarity matrix shape: {self.item_similarity.shape}")

    def _fit_svd(self):
        """Fit SVD for matrix factorization."""
        # Auto-adjust n_factors if larger than matrix dimensions
        n_users, n_items = self.user_item_matrix.shape
        max_factors = min(n_users, n_items) - 1

        # Cap factors to encourage generalization rather than perfect reconstruction.
        # Using near-full rank causes SVD to memorize the sparse matrix,
        # yielding ~0 scores for unseen items. A lower rank captures latent
        # patterns and produces meaningful predictions for unseen items.
        generalization_cap = max(2, min(n_users, n_items) // 3)
        max_factors = min(max_factors, generalization_cap)
        actual_factors = min(self.n_factors, max_factors)

        if actual_factors < self.n_factors:
            print(f"Adjusting n_factors from {self.n_factors} to {actual_factors} (matrix size: {n_users}x{n_items})")
            self.n_factors = actual_factors

        print(f"Fitting SVD with {self.n_factors} factors...")
        self.svd_model = TruncatedSVD(n_components=self.n_factors, random_state=42)

        # Fit and transform
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd_model.components_.T

        explained_var = self.svd_model.explained_variance_ratio_.sum()
        print(f"SVD explained variance: {explained_var:.4f}")

    def recommend(self,
                  user_id: int,
                  n_recommendations: int = N_RECOMMENDATIONS,
                  exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User ID to get recommendations for
            n_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude already seen articles

        Returns:
            List of (article_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if user_id not in self.user_id_map:
            # Cold start: return popular items
            return self._popular_items(n_recommendations)

        user_idx = self.user_id_map[user_id]

        if self.method == 'user_based':
            scores = self._recommend_user_based(user_idx)
        elif self.method == 'item_based':
            scores = self._recommend_item_based(user_idx)
        elif self.method == 'svd':
            scores = self._recommend_svd(user_idx)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Get seen articles
        if exclude_seen:
            seen_mask = np.array(self.user_item_matrix[user_idx].todense()).flatten() > 0
            scores[seen_mask] = -np.inf

        # Get top N
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [
            (self.reverse_article_map[idx], float(scores[idx]))
            for idx in top_indices if scores[idx] > -np.inf
        ]

        return recommendations

    def _recommend_user_based(self, user_idx: int) -> np.ndarray:
        """Generate scores using user-based CF."""
        # Get similar users
        user_similarities = self.user_similarity[user_idx]

        # Get top N similar users (excluding self)
        similar_user_indices = np.argsort(user_similarities)[::-1][1:self.n_neighbors + 1]
        similar_user_weights = user_similarities[similar_user_indices]

        # Weighted sum of similar users' interactions
        similar_users_matrix = self.user_item_matrix[similar_user_indices].toarray()
        scores = np.dot(similar_user_weights, similar_users_matrix)

        # Normalize by sum of weights
        weight_sum = np.sum(similar_user_weights)
        if weight_sum > 0:
            scores /= weight_sum

        return scores

    def _recommend_item_based(self, user_idx: int) -> np.ndarray:
        """Generate scores using item-based CF."""
        # Get user's interactions
        user_interactions = np.array(self.user_item_matrix[user_idx].todense()).flatten()
        interacted_items = np.where(user_interactions > 0)[0]

        if len(interacted_items) == 0:
            return np.zeros(self.user_item_matrix.shape[1])

        # For each item, compute score based on similarity to interacted items
        scores = np.zeros(self.user_item_matrix.shape[1])

        for item_idx in range(len(scores)):
            # Get similarities to items user has interacted with
            item_sims = self.item_similarity[item_idx, interacted_items]

            # Take top K most similar items
            top_k = min(self.n_neighbors, len(item_sims))
            top_sims = np.partition(item_sims, -top_k)[-top_k:]

            scores[item_idx] = np.mean(top_sims)

        return scores

    def _recommend_svd(self, user_idx: int) -> np.ndarray:
        """Generate scores using SVD matrix factorization."""
        # Predict scores as dot product of user and item factors
        user_vector = self.user_factors[user_idx]
        scores = np.dot(user_vector, self.item_factors.T)
        return scores

    def _popular_items(self, n: int) -> List[Tuple[int, float]]:
        """Return most popular items (cold start fallback)."""
        item_popularity = np.array(self.user_item_matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(item_popularity)[::-1][:n]

        return [
            (self.reverse_article_map[idx], float(item_popularity[idx]))
            for idx in top_indices
        ]

    def get_similar_users(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get most similar users to a given user.

        Args:
            user_id: User ID
            n: Number of similar users to return

        Returns:
            List of (user_id, similarity) tuples
        """
        if self.user_similarity is None:
            raise ValueError("User similarity not computed. Use method='user_based'.")

        if user_id not in self.user_id_map:
            return []

        user_idx = self.user_id_map[user_id]
        similarities = self.user_similarity[user_idx]

        top_indices = np.argsort(similarities)[::-1][1:n + 1]
        return [
            (self.reverse_user_map[idx], float(similarities[idx]))
            for idx in top_indices
        ]

    def get_similar_items(self, article_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get most similar items to a given article.

        Args:
            article_id: Article ID
            n: Number of similar items to return

        Returns:
            List of (article_id, similarity) tuples
        """
        if self.item_similarity is None:
            raise ValueError("Item similarity not computed. Use method='item_based'.")

        if article_id not in self.article_id_map:
            return []

        item_idx = self.article_id_map[article_id]
        similarities = self.item_similarity[item_idx]

        top_indices = np.argsort(similarities)[::-1][1:n + 1]
        return [
            (self.reverse_article_map[idx], float(similarities[idx]))
            for idx in top_indices
        ]

    def save(self, path: Optional[str] = None):
        """Save the model to disk."""
        if path is None:
            path = f"{MODEL_DIR}/cf_{self.method}_model.joblib"

        model_data = {
            'method': self.method,
            'n_neighbors': self.n_neighbors,
            'n_factors': self.n_factors,
            'user_item_matrix': self.user_item_matrix,
            'user_similarity': self.user_similarity,
            'item_similarity': self.item_similarity,
            'svd_model': self.svd_model,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_id_map': self.user_id_map,
            'article_id_map': self.article_id_map,
            'is_fitted': self.is_fitted
        }

        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'CollaborativeFiltering':
        """Load a model from disk."""
        model_data = joblib.load(path)

        model = cls(
            method=model_data['method'],
            n_neighbors=model_data['n_neighbors'],
            n_factors=model_data['n_factors']
        )

        model.user_item_matrix = model_data['user_item_matrix']
        model.user_similarity = model_data['user_similarity']
        model.item_similarity = model_data['item_similarity']
        model.svd_model = model_data['svd_model']
        model.user_factors = model_data['user_factors']
        model.item_factors = model_data['item_factors']
        model.user_id_map = model_data['user_id_map']
        model.article_id_map = model_data['article_id_map']
        model.reverse_user_map = {v: k for k, v in model.user_id_map.items()}
        model.reverse_article_map = {v: k for k, v in model.article_id_map.items()}
        model.is_fitted = model_data['is_fitted']

        print(f"Model loaded from {path}")
        return model

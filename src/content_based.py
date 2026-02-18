"""
Content-Based Recommender
Recommends articles based on content similarity using embeddings
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import joblib

from .config import CONTENT_PARAMS, N_RECOMMENDATIONS, MODEL_DIR


class ContentBasedRecommender:
    """
    Content-Based Recommender System.

    Uses article embeddings to find similar articles to what users have read.
    Supports embedding reduction via PCA for deployment efficiency.
    """

    def __init__(self,
                 embedding_dim: int = CONTENT_PARAMS['embedding_dim'],
                 use_pca: bool = True):
        """
        Initialize the content-based recommender.

        Args:
            embedding_dim: Target embedding dimension after PCA
            use_pca: Whether to use PCA for dimension reduction
        """
        self.embedding_dim = embedding_dim
        self.use_pca = use_pca

        # Model components
        self.embeddings = None  # Original embeddings
        self.reduced_embeddings = None  # PCA-reduced embeddings
        self.pca_model = None
        self.article_ids = None  # List of article IDs in order
        self.article_id_to_idx = None
        self.similarity_matrix = None

        # User profiles
        self.user_profiles = {}  # user_id -> embedding vector

        self.is_fitted = False

    def fit(self,
            embeddings: Dict[int, np.ndarray],
            user_interactions: Optional[Dict[int, List[int]]] = None) -> 'ContentBasedRecommender':
        """
        Fit the content-based recommender.

        Args:
            embeddings: Dictionary mapping article_id to embedding vector
            user_interactions: Optional dict mapping user_id to list of article_ids

        Returns:
            self for method chaining
        """
        self.embeddings = embeddings
        self.article_ids = list(embeddings.keys())
        self.article_id_to_idx = {aid: idx for idx, aid in enumerate(self.article_ids)}

        # Stack embeddings into matrix
        embedding_matrix = np.array([embeddings[aid] for aid in self.article_ids])
        print(f"Original embedding shape: {embedding_matrix.shape}")

        # Apply PCA if requested
        if self.use_pca and embedding_matrix.shape[1] > self.embedding_dim:
            print(f"Applying PCA to reduce to {self.embedding_dim} dimensions...")
            self.pca_model = PCA(n_components=self.embedding_dim, random_state=42)
            self.reduced_embeddings = self.pca_model.fit_transform(embedding_matrix)

            explained_var = self.pca_model.explained_variance_ratio_.sum()
            print(f"PCA explained variance: {explained_var:.4f}")
        else:
            self.reduced_embeddings = embedding_matrix

        # Normalize embeddings for cosine similarity
        self.reduced_embeddings = normalize(self.reduced_embeddings)

        # NOTE: We do NOT precompute the full similarity matrix (364k x 364k)
        # as it would require ~1 TB of RAM. Instead, similarity is computed
        # on-the-fly in recommend_similar_articles().
        self.similarity_matrix = None
        print(f"Embeddings ready: {self.reduced_embeddings.shape[0]} articles, {self.reduced_embeddings.shape[1]} dims")

        # Build user profiles if interactions provided
        if user_interactions:
            print("Building user profiles...")
            for user_id, articles in user_interactions.items():
                self._update_user_profile(user_id, articles)

        self.is_fitted = True
        return self

    def _update_user_profile(self, user_id: int, article_ids: List[int]):
        """
        Update user profile based on interacted articles.
        Profile is the mean of article embeddings.

        Args:
            user_id: User ID
            article_ids: List of article IDs the user has interacted with
        """
        valid_indices = [
            self.article_id_to_idx[aid]
            for aid in article_ids
            if aid in self.article_id_to_idx
        ]

        if len(valid_indices) > 0:
            profile = np.mean(self.reduced_embeddings[valid_indices], axis=0)
            self.user_profiles[user_id] = normalize(profile.reshape(1, -1)).flatten()

    def recommend(self,
                  user_id: int,
                  user_history: Optional[List[int]] = None,
                  n_recommendations: int = N_RECOMMENDATIONS,
                  exclude_seen: bool = True,
                  candidate_articles: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User ID to get recommendations for
            user_history: List of article IDs user has read (optional, uses stored profile)
            n_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude already seen articles
            candidate_articles: Optional list of article IDs to consider.
                              If None, all articles in the embedding space are candidates.

        Returns:
            List of (article_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Update profile if history provided
        if user_history:
            self._update_user_profile(user_id, user_history)

        # Get user profile
        if user_id not in self.user_profiles:
            # Cold start: return random diverse items
            return self._diverse_items(n_recommendations)

        user_profile = self.user_profiles[user_id]

        # If candidate set provided, only score those articles
        if candidate_articles is not None:
            candidate_indices = [
                self.article_id_to_idx[aid]
                for aid in candidate_articles
                if aid in self.article_id_to_idx
            ]
            if not candidate_indices:
                return self._diverse_items(n_recommendations)

            candidate_embeddings = self.reduced_embeddings[candidate_indices]
            scores = np.dot(candidate_embeddings, user_profile)

            # Exclude seen articles
            if exclude_seen and user_history:
                user_history_set = set(user_history)
                for i, idx in enumerate(candidate_indices):
                    if self.article_ids[idx] in user_history_set:
                        scores[i] = -np.inf

            top_local = np.argsort(scores)[::-1][:n_recommendations]
            recommendations = [
                (self.article_ids[candidate_indices[li]], float(scores[li]))
                for li in top_local if scores[li] > -np.inf
            ]
            return recommendations

        # Default: score all articles
        scores = np.dot(self.reduced_embeddings, user_profile)

        # Exclude seen articles
        if exclude_seen and user_history:
            for aid in user_history:
                if aid in self.article_id_to_idx:
                    scores[self.article_id_to_idx[aid]] = -np.inf

        # Get top N
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [
            (self.article_ids[idx], float(scores[idx]))
            for idx in top_indices if scores[idx] > -np.inf
        ]

        return recommendations

    def recommend_similar_articles(self,
                                   article_id: int,
                                   n: int = N_RECOMMENDATIONS) -> List[Tuple[int, float]]:
        """
        Get articles similar to a given article.

        Args:
            article_id: Article ID
            n: Number of similar articles to return

        Returns:
            List of (article_id, similarity) tuples
        """
        if article_id not in self.article_id_to_idx:
            return []

        idx = self.article_id_to_idx[article_id]

        # Compute similarity for this single article against all others
        article_emb = self.reduced_embeddings[idx].reshape(1, -1)
        similarities = cosine_similarity(article_emb, self.reduced_embeddings).flatten()

        # Exclude self
        similarities[idx] = -np.inf

        top_indices = np.argsort(similarities)[::-1][:n]
        return [
            (self.article_ids[i], float(similarities[i]))
            for i in top_indices
        ]

    def _diverse_items(self, n: int) -> List[Tuple[int, float]]:
        """
        Return diverse items for cold start.
        Uses k-means clustering to select diverse articles.
        """
        from sklearn.cluster import KMeans

        # Cluster articles
        n_clusters = min(n, len(self.article_ids))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.reduced_embeddings)

        # Select one article from each cluster (closest to centroid)
        selected = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Find article closest to centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(
                self.reduced_embeddings[cluster_indices] - centroid, axis=1
            )
            closest_idx = cluster_indices[np.argmin(distances)]
            selected.append((self.article_ids[closest_idx], 1.0 / (len(selected) + 1)))

        return selected[:n]

    def get_user_profile(self, user_id: int) -> Optional[np.ndarray]:
        """Get the profile vector for a user."""
        return self.user_profiles.get(user_id)

    def get_article_embedding(self, article_id: int) -> Optional[np.ndarray]:
        """Get the (reduced) embedding for an article."""
        if article_id not in self.article_id_to_idx:
            return None
        idx = self.article_id_to_idx[article_id]
        return self.reduced_embeddings[idx]

    def save(self, path: Optional[str] = None):
        """Save the model to disk."""
        if path is None:
            path = f"{MODEL_DIR}/content_based_model.joblib"

        model_data = {
            'embedding_dim': self.embedding_dim,
            'use_pca': self.use_pca,
            'reduced_embeddings': self.reduced_embeddings,
            'pca_model': self.pca_model,
            'article_ids': self.article_ids,
            'article_id_to_idx': self.article_id_to_idx,
            'user_profiles': self.user_profiles,
            'is_fitted': self.is_fitted
        }

        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ContentBasedRecommender':
        """Load a model from disk."""
        model_data = joblib.load(path)

        model = cls(
            embedding_dim=model_data['embedding_dim'],
            use_pca=model_data['use_pca']
        )

        model.reduced_embeddings = model_data['reduced_embeddings']
        model.pca_model = model_data['pca_model']
        model.article_ids = model_data['article_ids']
        model.article_id_to_idx = model_data['article_id_to_idx']
        model.similarity_matrix = None
        model.user_profiles = model_data['user_profiles']
        model.is_fitted = model_data['is_fitted']

        print(f"Model loaded from {path}")
        return model

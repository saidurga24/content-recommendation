"""
Hybrid Recommender
Combines Collaborative Filtering and Content-Based approaches
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import joblib

from .config import HYBRID_WEIGHTS, N_RECOMMENDATIONS, MODEL_DIR
from .collaborative_filtering import CollaborativeFiltering
from .content_based import ContentBasedRecommender


class HybridRecommender:
    """
    Hybrid Recommender System.

    Combines collaborative filtering and content-based recommendations
    using weighted averaging or switching strategies.
    """

    def __init__(self,
                 cf_weight: float = HYBRID_WEIGHTS['collaborative'],
                 content_weight: float = HYBRID_WEIGHTS['content'],
                 strategy: str = 'weighted'):
        """
        Initialize the hybrid recommender.

        Args:
            cf_weight: Weight for collaborative filtering scores
            content_weight: Weight for content-based scores
            strategy: 'weighted', 'switching', or 'cascade'
        """
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.strategy = strategy

        # Component models
        self.cf_model: Optional[CollaborativeFiltering] = None
        self.content_model: Optional[ContentBasedRecommender] = None

        # Shared ID mappings
        self.article_ids = None
        self.user_id_map = None

        self.is_fitted = False

    def fit(self,
            cf_model: CollaborativeFiltering,
            content_model: ContentBasedRecommender) -> 'HybridRecommender':
        """
        Fit the hybrid model with pre-trained component models.

        Args:
            cf_model: Trained collaborative filtering model
            content_model: Trained content-based model

        Returns:
            self for method chaining
        """
        self.cf_model = cf_model
        self.content_model = content_model

        # Get shared article IDs
        cf_articles = set(cf_model.article_id_map.keys())
        content_articles = set(content_model.article_ids)
        self.article_ids = list(cf_articles & content_articles)

        self.user_id_map = cf_model.user_id_map

        self.is_fitted = True
        print(f"Hybrid model fitted with {len(self.article_ids)} shared articles")
        return self

    def recommend(self,
                  user_id: int,
                  user_history: Optional[List[int]] = None,
                  n_recommendations: int = N_RECOMMENDATIONS,
                  exclude_seen: bool = True,
                  candidate_articles: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate hybrid recommendations for a user.

        Args:
            user_id: User ID to get recommendations for
            user_history: List of article IDs user has read
            n_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude already seen articles
            candidate_articles: Optional list of article IDs to consider

        Returns:
            List of (article_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.strategy == 'weighted':
            return self._weighted_recommend(
                user_id, user_history, n_recommendations, exclude_seen,
                candidate_articles
            )
        elif self.strategy == 'switching':
            return self._switching_recommend(
                user_id, user_history, n_recommendations, exclude_seen
            )
        elif self.strategy == 'cascade':
            return self._cascade_recommend(
                user_id, user_history, n_recommendations, exclude_seen
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_recommend(self,
                            user_id: int,
                            user_history: Optional[List[int]],
                            n_recommendations: int,
                            exclude_seen: bool,
                            candidate_articles: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Weighted combination of CF and content-based scores.
        """
        # Get more recommendations than needed for merging
        n_fetch = n_recommendations * 3

        # Get CF recommendations
        cf_recs = self.cf_model.recommend(
            user_id, n_recommendations=n_fetch, exclude_seen=exclude_seen
        )
        cf_scores = {aid: score for aid, score in cf_recs}

        # Get content-based recommendations (pass candidate_articles if available)
        content_kwargs = {
            'user_history': user_history,
            'n_recommendations': n_fetch,
            'exclude_seen': exclude_seen,
        }
        if candidate_articles is not None:
            content_kwargs['candidate_articles'] = candidate_articles
        content_recs = self.content_model.recommend(user_id, **content_kwargs)
        content_scores = {aid: score for aid, score in content_recs}

        # Normalize scores
        cf_scores = self._normalize_scores(cf_scores)
        content_scores = self._normalize_scores(content_scores)

        # Combine scores
        all_articles = set(cf_scores.keys()) | set(content_scores.keys())
        combined_scores = {}

        for aid in all_articles:
            cf_score = cf_scores.get(aid, 0)
            content_score = content_scores.get(aid, 0)
            combined_scores[aid] = (
                self.cf_weight * cf_score +
                self.content_weight * content_score
            )

        # Sort and return top N
        sorted_items = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_items[:n_recommendations]

    def _switching_recommend(self,
                             user_id: int,
                             user_history: Optional[List[int]],
                             n_recommendations: int,
                             exclude_seen: bool) -> List[Tuple[int, float]]:
        """
        Switch between CF and content-based based on user profile.
        Use CF for users with many interactions, content for new users.
        """
        # Check if user has enough interactions for CF
        min_interactions = 5

        if user_id in self.user_id_map:
            user_idx = self.user_id_map[user_id]
            n_interactions = self.cf_model.user_item_matrix[user_idx].nnz

            if n_interactions >= min_interactions:
                # Use CF for active users
                return self.cf_model.recommend(
                    user_id, n_recommendations=n_recommendations,
                    exclude_seen=exclude_seen
                )

        # Use content-based for new/inactive users
        return self.content_model.recommend(
            user_id, user_history=user_history,
            n_recommendations=n_recommendations,
            exclude_seen=exclude_seen
        )

    def _cascade_recommend(self,
                           user_id: int,
                           user_history: Optional[List[int]],
                           n_recommendations: int,
                           exclude_seen: bool) -> List[Tuple[int, float]]:
        """
        Cascade approach: use CF first, then refine with content-based.
        """
        # Get more CF recommendations
        n_candidates = n_recommendations * 5
        cf_recs = self.cf_model.recommend(
            user_id, n_recommendations=n_candidates, exclude_seen=exclude_seen
        )

        if len(cf_recs) == 0:
            return self.content_model.recommend(
                user_id, user_history=user_history,
                n_recommendations=n_recommendations,
                exclude_seen=exclude_seen
            )

        # Refine with content-based scores
        candidate_ids = [aid for aid, _ in cf_recs]

        # Get content scores for candidates
        refined_scores = []
        for aid in candidate_ids:
            content_similar = self.content_model.recommend_similar_articles(aid, n=1)
            content_score = content_similar[0][1] if content_similar else 0

            cf_score = dict(cf_recs).get(aid, 0)
            combined = self.cf_weight * cf_score + self.content_weight * content_score
            refined_scores.append((aid, combined))

        # Sort by combined score
        refined_scores.sort(key=lambda x: x[1], reverse=True)
        return refined_scores[:n_recommendations]

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return {}

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return {k: 0.5 for k in scores}

        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }

    def save(self, path: Optional[str] = None):
        """Save the hybrid model configuration."""
        if path is None:
            path = f"{MODEL_DIR}/hybrid_model.joblib"

        model_data = {
            'cf_weight': self.cf_weight,
            'content_weight': self.content_weight,
            'strategy': self.strategy,
            'article_ids': self.article_ids,
            'is_fitted': self.is_fitted
        }

        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model_data, path)
        print(f"Hybrid config saved to {path}")

    @classmethod
    def load(cls,
             path: str,
             cf_model: CollaborativeFiltering,
             content_model: ContentBasedRecommender) -> 'HybridRecommender':
        """
        Load a hybrid model from disk.

        Args:
            path: Path to saved model
            cf_model: Loaded CF model
            content_model: Loaded content-based model
        """
        model_data = joblib.load(path)

        model = cls(
            cf_weight=model_data['cf_weight'],
            content_weight=model_data['content_weight'],
            strategy=model_data['strategy']
        )

        model.cf_model = cf_model
        model.content_model = content_model
        model.article_ids = model_data['article_ids']
        model.user_id_map = cf_model.user_id_map
        model.is_fitted = model_data['is_fitted']

        print(f"Hybrid model loaded from {path}")
        return model

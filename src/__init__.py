# My Content Recommendation System
# Core module for recommendation algorithms

from .config import *
from .data_loader import DataLoader
from .collaborative_filtering import CollaborativeFiltering
from .content_based import ContentBasedRecommender
from .hybrid_recommender import HybridRecommender

# Import only the non-visualization utilities eagerly.
# Visualization helpers (which require matplotlib/seaborn) are available
# via explicit import: from src.utils import plot_model_comparison, ...
from .utils import (
    precision_at_k, recall_at_k, f1_at_k, ndcg_at_k,
    mean_reciprocal_rank, hit_rate_at_k, coverage, diversity,
    evaluate_model, get_popular_items, get_active_users,
    create_user_item_dict, sample_negative_items,
)

__version__ = "1.0.0"
__author__ = "Sai Durga Prasad"

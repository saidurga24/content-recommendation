"""
Utility functions for the Content Recommendation System
Includes evaluation metrics, visualization, and helper functions
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Optional


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Compute Precision@K.

    Args:
        recommended: List of recommended item IDs (in order)
        relevant: Set of relevant (ground truth) item IDs
        k: Number of top recommendations to consider

    Returns:
        Precision@K score
    """
    if k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    n_relevant = len(set(recommended_k) & relevant)
    return n_relevant / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Compute Recall@K.

    Args:
        recommended: List of recommended item IDs (in order)
        relevant: Set of relevant (ground truth) item IDs
        k: Number of top recommendations to consider

    Returns:
        Recall@K score
    """
    if len(relevant) == 0:
        return 0.0

    recommended_k = recommended[:k]
    n_relevant = len(set(recommended_k) & relevant)
    return n_relevant / len(relevant)


def f1_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Compute F1@K score.

    Args:
        recommended: List of recommended item IDs (in order)
        relevant: Set of relevant (ground truth) item IDs
        k: Number of top recommendations to consider

    Returns:
        F1@K score
    """
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Args:
        recommended: List of recommended item IDs (in order)
        relevant: Set of relevant (ground truth) item IDs
        k: Number of top recommendations to consider

    Returns:
        NDCG@K score
    """
    recommended_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because positions are 1-indexed

    # IDCG (ideal DCG)
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_reciprocal_rank(recommended: List[int], relevant: Set[int]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        recommended: List of recommended item IDs (in order)
        relevant: Set of relevant (ground truth) item IDs

    Returns:
        MRR score
    """
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Compute Hit Rate at K (1 if any relevant item in top K, 0 otherwise).

    Args:
        recommended: List of recommended item IDs (in order)
        relevant: Set of relevant (ground truth) item IDs
        k: Number of top recommendations to consider

    Returns:
        Hit rate (0 or 1)
    """
    recommended_k = set(recommended[:k])
    return 1.0 if len(recommended_k & relevant) > 0 else 0.0


def coverage(all_recommendations: List[List[int]], all_items: Set[int]) -> float:
    """
    Compute catalog coverage - fraction of items ever recommended.

    Args:
        all_recommendations: List of recommendation lists for all users
        all_items: Set of all item IDs in catalog

    Returns:
        Coverage score
    """
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)

    return len(recommended_items) / len(all_items) if all_items else 0.0


def diversity(recommendations: List[int], similarity_matrix: np.ndarray,
              item_to_idx: Dict[int, int]) -> float:
    """
    Compute intra-list diversity (1 - average pairwise similarity).

    Args:
        recommendations: List of recommended item IDs
        similarity_matrix: Item-item similarity matrix
        item_to_idx: Mapping from item ID to matrix index

    Returns:
        Diversity score
    """
    if len(recommendations) < 2:
        return 1.0

    indices = [item_to_idx[item] for item in recommendations if item in item_to_idx]

    if len(indices) < 2:
        return 1.0

    similarities = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            similarities.append(similarity_matrix[indices[i], indices[j]])

    avg_similarity = np.mean(similarities)
    return 1 - avg_similarity


def evaluate_model(model,
                   test_data: pd.DataFrame,
                   user_col: str,
                   article_col: str,
                   k: int = 5,
                   train_data: pd.DataFrame = None) -> Dict[str, float]:
    """
    Evaluate a recommendation model on test data.

    Args:
        model: Recommender model with recommend() method
        test_data: Test DataFrame with user-item interactions
        user_col: Name of user ID column
        article_col: Name of article ID column
        k: Number of recommendations to evaluate
        train_data: Optional training DataFrame (used to provide user_history
                    and candidate_articles to content-based/hybrid models)

    Returns:
        Dictionary with evaluation metrics
    """
    # Group test data by user
    user_relevant = test_data.groupby(user_col)[article_col].apply(set).to_dict()

    # Build per-user training history and candidate article list
    if train_data is not None:
        user_train_history = train_data.groupby(user_col)[article_col].apply(list).to_dict()
        all_articles = list(
            set(train_data[article_col].unique()) | set(test_data[article_col].unique())
        )
    else:
        user_train_history = {}
        all_articles = None

    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'ndcg': [],
        'mrr': [],
        'hit_rate': []
    }

    # Check what keyword arguments the model's recommend() accepts
    import inspect
    rec_sig = inspect.signature(model.recommend)
    rec_params = set(rec_sig.parameters.keys())

    for user_id, relevant in user_relevant.items():
        try:
            # Build kwargs dynamically based on what the model accepts
            kwargs = {'n_recommendations': k}
            if 'user_history' in rec_params and user_id in user_train_history:
                kwargs['user_history'] = user_train_history[user_id]
            if 'candidate_articles' in rec_params and all_articles is not None:
                kwargs['candidate_articles'] = all_articles

            # Get recommendations
            recs = model.recommend(user_id, **kwargs)
            rec_ids = [item_id for item_id, _ in recs]

            # Compute metrics
            metrics['precision'].append(precision_at_k(rec_ids, relevant, k))
            metrics['recall'].append(recall_at_k(rec_ids, relevant, k))
            metrics['f1'].append(f1_at_k(rec_ids, relevant, k))
            metrics['ndcg'].append(ndcg_at_k(rec_ids, relevant, k))
            metrics['mrr'].append(mean_reciprocal_rank(rec_ids, relevant))
            metrics['hit_rate'].append(hit_rate_at_k(rec_ids, relevant, k))
        except Exception as e:
            # Skip users that cause errors (e.g., not in training data)
            continue

    # Average metrics
    return {
        f'precision@{k}': np.mean(metrics['precision']) if metrics['precision'] else 0.0,
        f'recall@{k}': np.mean(metrics['recall']) if metrics['recall'] else 0.0,
        f'f1@{k}': np.mean(metrics['f1']) if metrics['f1'] else 0.0,
        f'ndcg@{k}': np.mean(metrics['ndcg']) if metrics['ndcg'] else 0.0,
        'mrr': np.mean(metrics['mrr']) if metrics['mrr'] else 0.0,
        f'hit_rate@{k}': np.mean(metrics['hit_rate']) if metrics['hit_rate'] else 0.0
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_interaction_distribution(df: pd.DataFrame,
                                  user_col: str,
                                  article_col: str,
                                  figsize: Tuple[int, int] = (14, 5)):
    """
    Plot distribution of user and item interactions.

    Args:
        df: DataFrame with interactions
        user_col: Name of user ID column
        article_col: Name of article ID column
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # User interaction distribution
    user_counts = df[user_col].value_counts()
    axes[0].hist(user_counts, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Interactions')
    axes[0].set_ylabel('Number of Users')
    axes[0].set_title('Distribution of Interactions per User')
    axes[0].axvline(user_counts.median(), color='red', linestyle='--',
                    label=f'Median: {user_counts.median():.0f}')
    axes[0].legend()

    # Article interaction distribution
    article_counts = df[article_col].value_counts()
    axes[1].hist(article_counts, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Number of Interactions')
    axes[1].set_ylabel('Number of Articles')
    axes[1].set_title('Distribution of Interactions per Article')
    axes[1].axvline(article_counts.median(), color='red', linestyle='--',
                    label=f'Median: {article_counts.median():.0f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('interaction_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]],
                          metrics: List[str] = None,
                          figsize: Tuple[int, int] = (12, 6)):
    """
    Plot comparison of different models.

    Args:
        results: Dictionary mapping model names to metric dictionaries
        metrics: List of metrics to plot (default: all)
        figsize: Figure size
    """
    if metrics is None:
        metrics = list(list(results.values())[0].keys())

    models = list(results.keys())
    n_metrics = len(metrics)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]
        bars = axes[i].bar(models, values, color=colors)
        axes[i].set_title(metric)
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_similarity_heatmap(similarity_matrix: np.ndarray,
                            labels: List[str] = None,
                            title: str = 'Similarity Matrix',
                            figsize: Tuple[int, int] = (10, 8)):
    """
    Plot heatmap of similarity matrix (subset if too large).

    Args:
        similarity_matrix: Similarity matrix
        labels: Labels for rows/columns
        title: Plot title
        figsize: Figure size
    """
    # Limit size for visualization
    max_size = 50
    n = min(similarity_matrix.shape[0], max_size)
    matrix_subset = similarity_matrix[:n, :n]

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=figsize)
    sns.heatmap(matrix_subset, cmap='coolwarm', center=0,
                xticklabels=False, yticklabels=False)
    plt.title(f'{title} (showing {n}x{n} subset)')
    plt.savefig('similarity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_embedding_tsne(embeddings: np.ndarray,
                        labels: List[int] = None,
                        perplexity: int = 30,
                        figsize: Tuple[int, int] = (10, 10)):
    """
    Visualize embeddings using t-SNE.

    Args:
        embeddings: Embedding matrix
        labels: Optional labels for coloring
        perplexity: t-SNE perplexity parameter
        figsize: Figure size
    """
    from sklearn.manifold import TSNE

    # Limit to subset for speed
    max_points = 1000
    n = min(len(embeddings), max_points)
    indices = np.random.choice(len(embeddings), n, replace=False)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings[indices])

    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    if labels is not None:
        labels_subset = [labels[i] for i in indices]
        plt.scatter(coords[:, 0], coords[:, 1], c=labels_subset, alpha=0.6, cmap='tab10')
        plt.colorbar(label='Cluster')
    else:
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)

    plt.title(f't-SNE Visualization of Article Embeddings (n={n})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('embedding_tsne.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_popular_items(df: pd.DataFrame,
                      article_col: str,
                      n: int = 10) -> List[Tuple[int, int]]:
    """
    Get most popular items by interaction count.

    Args:
        df: DataFrame with interactions
        article_col: Name of article ID column
        n: Number of items to return

    Returns:
        List of (article_id, count) tuples
    """
    counts = df[article_col].value_counts().head(n)
    return list(counts.items())


def get_active_users(df: pd.DataFrame,
                     user_col: str,
                     n: int = 10) -> List[Tuple[int, int]]:
    """
    Get most active users by interaction count.

    Args:
        df: DataFrame with interactions
        user_col: Name of user ID column
        n: Number of users to return

    Returns:
        List of (user_id, count) tuples
    """
    counts = df[user_col].value_counts().head(n)
    return list(counts.items())


def create_user_item_dict(df: pd.DataFrame,
                          user_col: str,
                          article_col: str) -> Dict[int, List[int]]:
    """
    Create dictionary mapping users to their interacted items.

    Args:
        df: DataFrame with interactions
        user_col: Name of user ID column
        article_col: Name of article ID column

    Returns:
        Dictionary mapping user_id to list of article_ids
    """
    return df.groupby(user_col)[article_col].apply(list).to_dict()


def sample_negative_items(user_items: List[int],
                          all_items: List[int],
                          n_samples: int) -> List[int]:
    """
    Sample negative items (items user hasn't interacted with).

    Args:
        user_items: List of items user has interacted with
        all_items: List of all available items
        n_samples: Number of negative samples

    Returns:
        List of negative item IDs
    """
    user_items_set = set(user_items)
    negative_pool = [item for item in all_items if item not in user_items_set]

    n_samples = min(n_samples, len(negative_pool))
    return list(np.random.choice(negative_pool, n_samples, replace=False))

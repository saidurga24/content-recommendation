"""
Data Loader for the Content Recommendation System
Handles loading, preprocessing, and splitting of interaction data
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

from .config import (
    DATA_DIR, CLICKS_FILE, ARTICLES_FILE, EMBEDDINGS_FILE,
    TEST_SIZE, RANDOM_STATE, MIN_USER_INTERACTIONS, MIN_ARTICLE_INTERACTIONS
)


class DataLoader:
    """
    Data loader class for handling user-article interaction data.

    Attributes:
        clicks_df: DataFrame with user clicks/interactions
        articles_df: DataFrame with article metadata
        embeddings: Article embeddings dictionary
        user_item_matrix: Sparse user-item interaction matrix
    """

    def __init__(self, data_dir: str = DATA_DIR):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = data_dir
        self.clicks_df = None
        self.articles_df = None
        self.embeddings = None
        self.embeddings_array = None  # Original numpy array if loaded from array format
        self.user_item_matrix = None
        self.user_id_map = None  # Maps original user IDs to matrix indices
        self.article_id_map = None  # Maps original article IDs to matrix indices
        self.reverse_user_map = None  # Maps matrix indices to user IDs
        self.reverse_article_map = None  # Maps matrix indices to article IDs

    def load_data(self,
                  clicks_path: Optional[str] = None,
                  articles_path: Optional[str] = None,
                  embeddings_path: Optional[str] = None) -> 'DataLoader':
        """
        Load all data files.

        Args:
            clicks_path: Path to clicks CSV file
            articles_path: Path to articles metadata CSV file
            embeddings_path: Path to embeddings pickle file

        Returns:
            self for method chaining
        """
        # Load clicks data
        clicks_file = clicks_path or CLICKS_FILE
        if os.path.exists(clicks_file):
            self.clicks_df = pd.read_csv(clicks_file)
            print(f"Loaded clicks data: {len(self.clicks_df)} interactions")
        else:
            print(f"Warning: Clicks file not found at {clicks_file}")

        # Load articles metadata
        articles_file = articles_path or ARTICLES_FILE
        if os.path.exists(articles_file):
            self.articles_df = pd.read_csv(articles_file)
            print(f"Loaded articles data: {len(self.articles_df)} articles")
        else:
            print(f"Warning: Articles file not found at {articles_file}")

        # Load embeddings
        embeddings_file = embeddings_path or EMBEDDINGS_FILE
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                raw_embeddings = pickle.load(f)

            # Handle both numpy array and dictionary formats
            if isinstance(raw_embeddings, np.ndarray):
                # Convert numpy array to dictionary {index: embedding}
                self.embeddings = {i: raw_embeddings[i] for i in range(len(raw_embeddings))}
                self.embeddings_array = raw_embeddings  # Keep original array for efficiency
                print(f"Loaded embeddings: {len(self.embeddings)} articles (shape: {raw_embeddings.shape})")
            else:
                # Already a dictionary
                self.embeddings = raw_embeddings
                self.embeddings_array = None
                print(f"Loaded embeddings: {len(self.embeddings)} articles")
        else:
            print(f"Warning: Embeddings file not found at {embeddings_file}")

        return self

    def preprocess(self,
                   min_user_interactions: int = MIN_USER_INTERACTIONS,
                   min_article_interactions: int = MIN_ARTICLE_INTERACTIONS) -> 'DataLoader':
        """
        Preprocess the data by filtering users and articles with few interactions.

        Args:
            min_user_interactions: Minimum clicks for a user to be included
            min_article_interactions: Minimum clicks for an article to be included

        Returns:
            self for method chaining
        """
        if self.clicks_df is None:
            raise ValueError("Clicks data not loaded. Call load_data() first.")

        original_len = len(self.clicks_df)

        # Identify user and article columns
        user_col = self._get_user_column()
        article_col = self._get_article_column()

        # Filter users with minimum interactions
        user_counts = self.clicks_df[user_col].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        self.clicks_df = self.clicks_df[self.clicks_df[user_col].isin(valid_users)]

        # Filter articles with minimum interactions
        article_counts = self.clicks_df[article_col].value_counts()
        valid_articles = article_counts[article_counts >= min_article_interactions].index
        self.clicks_df = self.clicks_df[self.clicks_df[article_col].isin(valid_articles)]

        print(f"Preprocessed data: {original_len} -> {len(self.clicks_df)} interactions")
        print(f"Unique users: {self.clicks_df[user_col].nunique()}")
        print(f"Unique articles: {self.clicks_df[article_col].nunique()}")

        return self

    def _get_user_column(self) -> str:
        """Get the user ID column name from the clicks dataframe."""
        possible_names = ['user_id', 'userId', 'user', 'session_id', 'sessionId']
        for name in possible_names:
            if name in self.clicks_df.columns:
                return name
        raise ValueError(f"Could not find user column. Columns: {self.clicks_df.columns.tolist()}")

    def _get_article_column(self) -> str:
        """Get the article ID column name from the clicks dataframe."""
        possible_names = ['article_id', 'articleId', 'click_article_id', 'item_id', 'itemId']
        for name in possible_names:
            if name in self.clicks_df.columns:
                return name
        raise ValueError(f"Could not find article column. Columns: {self.clicks_df.columns.tolist()}")

    def build_user_item_matrix(self, data: pd.DataFrame = None) -> csr_matrix:
        """
        Build a sparse user-item interaction matrix.

        Args:
            data: Optional DataFrame to build matrix from.
                  If None, uses self.clicks_df (all interactions).
                  Pass train_df to build from training data only.

        Returns:
            Sparse CSR matrix of shape (n_users, n_articles)
        """
        if data is None:
            if self.clicks_df is None:
                raise ValueError("Clicks data not loaded. Call load_data() first.")
            data = self.clicks_df

        user_col = self._get_user_column()
        article_col = self._get_article_column()

        # Create ID mappings from ALL preprocessed data (ensures consistent indices)
        all_users = self.clicks_df[user_col].unique()
        all_articles = self.clicks_df[article_col].unique()

        self.user_id_map = {uid: idx for idx, uid in enumerate(all_users)}
        self.article_id_map = {aid: idx for idx, aid in enumerate(all_articles)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_article_map = {idx: aid for aid, idx in self.article_id_map.items()}

        # Build sparse matrix from the provided data
        row_indices = data[user_col].map(self.user_id_map).values
        col_indices = data[article_col].map(self.article_id_map).values
        values = np.ones(len(data))  # Binary interactions

        self.user_item_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(all_users), len(all_articles))
        )

        print(f"Built user-item matrix: {self.user_item_matrix.shape}")
        print(f"Sparsity: {1 - self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape):.4f}")

        return self.user_item_matrix

    def train_test_split(self,
                         test_size: float = TEST_SIZE,
                         random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the interactions into train and test sets.
        Uses time-based or random split depending on data.

        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, test_df)
        """
        if self.clicks_df is None:
            raise ValueError("Clicks data not loaded. Call load_data() first.")

        user_col = self._get_user_column()

        # Check if timestamp column exists for time-based split
        time_cols = ['timestamp', 'click_timestamp', 'session_start', 'time']
        time_col = None
        for col in time_cols:
            if col in self.clicks_df.columns:
                time_col = col
                break

        if time_col:
            # Time-based split: keep last interaction(s) per user for test
            self.clicks_df = self.clicks_df.sort_values(time_col)

            train_data = []
            test_data = []

            for user_id, group in self.clicks_df.groupby(user_col):
                n_test = max(1, int(len(group) * test_size))
                train_data.append(group.iloc[:-n_test])
                test_data.append(group.iloc[-n_test:])

            train_df = pd.concat(train_data, ignore_index=True)
            test_df = pd.concat(test_data, ignore_index=True)
        else:
            # Random split
            train_df, test_df = train_test_split(
                self.clicks_df,
                test_size=test_size,
                random_state=random_state
            )

        print(f"Train set: {len(train_df)} interactions")
        print(f"Test set: {len(test_df)} interactions")

        return train_df, test_df

    def get_user_interactions(self, user_id: int) -> List[int]:
        """
        Get all articles a user has interacted with.

        Args:
            user_id: User ID

        Returns:
            List of article IDs
        """
        if self.clicks_df is None:
            raise ValueError("Clicks data not loaded.")

        user_col = self._get_user_column()
        article_col = self._get_article_column()

        interactions = self.clicks_df[self.clicks_df[user_col] == user_id][article_col].tolist()
        return interactions

    def get_article_info(self, article_id: int) -> Optional[Dict]:
        """
        Get metadata for an article.

        Args:
            article_id: Article ID

        Returns:
            Dictionary with article metadata or None
        """
        if self.articles_df is None:
            return None

        article_col = self._get_article_column_metadata()
        article = self.articles_df[self.articles_df[article_col] == article_id]

        if len(article) == 0:
            return None

        return article.iloc[0].to_dict()

    def _get_article_column_metadata(self) -> str:
        """Get the article ID column name from the articles metadata dataframe."""
        possible_names = ['article_id', 'articleId', 'id', 'item_id']
        for name in possible_names:
            if name in self.articles_df.columns:
                return name
        raise ValueError(f"Could not find article column in metadata. Columns: {self.articles_df.columns.tolist()}")

    def get_all_users(self) -> List[int]:
        """Get list of all user IDs."""
        if self.clicks_df is None:
            return []
        user_col = self._get_user_column()
        return self.clicks_df[user_col].unique().tolist()

    def get_all_articles(self) -> List[int]:
        """Get list of all article IDs."""
        if self.clicks_df is None:
            return []
        article_col = self._get_article_column()
        return self.clicks_df[article_col].unique().tolist()

    def get_article_embedding(self, article_id: int) -> Optional[np.ndarray]:
        """
        Get the embedding vector for an article.

        Args:
            article_id: Article ID

        Returns:
            Embedding vector or None if not found
        """
        if self.embeddings is None:
            return None

        # Use array indexing if available (faster)
        if hasattr(self, 'embeddings_array') and self.embeddings_array is not None:
            if 0 <= article_id < len(self.embeddings_array):
                return self.embeddings_array[article_id]
            return None

        return self.embeddings.get(article_id)

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        if self.clicks_df is None:
            return {}

        user_col = self._get_user_column()
        article_col = self._get_article_column()

        user_interactions = self.clicks_df[user_col].value_counts()
        article_interactions = self.clicks_df[article_col].value_counts()

        stats = {
            'n_interactions': len(self.clicks_df),
            'n_users': self.clicks_df[user_col].nunique(),
            'n_articles': self.clicks_df[article_col].nunique(),
            'avg_interactions_per_user': user_interactions.mean(),
            'median_interactions_per_user': user_interactions.median(),
            'avg_interactions_per_article': article_interactions.mean(),
            'median_interactions_per_article': article_interactions.median(),
        }

        if self.user_item_matrix is not None:
            stats['sparsity'] = 1 - self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape)

        return stats

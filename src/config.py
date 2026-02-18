"""
Configuration file for the Content Recommendation System
Centralizes all paths, parameters, and constants
"""

import os

# =============================================================================
# PATHS
# =============================================================================

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'news-portal-user-interactions-by-globocom')

# Model directory
MODEL_DIR = os.path.join(PROJECT_ROOT, 'P9_02_azure_function', 'model')

# Data file paths
CLICKS_FILE = os.path.join(DATA_DIR, 'clicks_full.csv')  # Use full dataset (3M interactions)
ARTICLES_FILE = os.path.join(DATA_DIR, 'articles_metadata.csv')
EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'articles_embeddings.pickle')

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# Number of recommendations to return
N_RECOMMENDATIONS = 5

# Collaborative Filtering parameters
CF_PARAMS = {
    'n_neighbors': 20,           # Number of similar users/items to consider
    'min_support': 3,            # Minimum number of common items for similarity
    'similarity_metric': 'cosine',  # cosine, pearson, jaccard
}

# Matrix Factorization (SVD) parameters
SVD_PARAMS = {
    'n_factors': 50,             # Number of latent factors
    'n_epochs': 20,              # Number of training epochs
    'lr': 0.005,                 # Learning rate
    'reg': 0.02,                 # Regularization term
}

# Content-Based parameters
CONTENT_PARAMS = {
    'embedding_dim': 250,        # Embedding dimension (after PCA reduction)
    'similarity_threshold': 0.1, # Minimum similarity to consider
}

# Hybrid model weights
HYBRID_WEIGHTS = {
    'collaborative': 0.6,        # Weight for CF score
    'content': 0.4,              # Weight for content-based score
}

# =============================================================================
# DATA PARAMETERS
# =============================================================================

# Train/test split ratio
TEST_SIZE = 0.2

# Random seed for reproducibility
RANDOM_STATE = 42

# Minimum interactions for a user to be included
MIN_USER_INTERACTIONS = 5

# Minimum interactions for an article to be included
MIN_ARTICLE_INTERACTIONS = 3

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Flask API settings
API_HOST = '0.0.0.0'
API_PORT = 5001

# Azure Function settings
AZURE_FUNCTION_URL = os.environ.get('AZURE_FUNCTION_URL', 'http://localhost:7071')

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

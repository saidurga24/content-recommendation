# Technical Note — Project 9: Content Recommendation System

**Project:** Create an Application for Recommending Content  
**Author:** Sai Durga Prasad  
**Program:** AI Engineer — OpenClassrooms  
**Date:** February 2026  

---

## 1. Executive Summary

This document presents the technical architecture and functional design of "My Content," a content recommendation system that delivers personalized article recommendations to users. The system implements three distinct recommendation approaches — Collaborative Filtering, Content-Based Filtering, and a Hybrid method — deployed as a serverless Azure Function behind a Flask web application.

The MVP accepts a user ID as input and returns 5 recommended articles ranked by relevance score. The system was evaluated using standard information retrieval metrics (Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR). Item-Based Collaborative Filtering achieved the best overall performance with 42.55% Hit Rate@5 and 0.0894 Precision@5.

---

## 2. Business Context

### 2.1 Problem Statement

"My Content" is a startup that aims to encourage reading by recommending relevant content (articles, books) to its users. The company needs an MVP recommendation system that can:

- Recommend 5 relevant articles to each user
- Support multiple recommendation strategies
- Be deployed as a scalable serverless application
- Handle the addition of new users and new articles over time

### 2.2 User Story

> "As a user of the application, I will receive a selection of five articles."

### 2.3 Dataset

The system uses the **Globo.com News Portal User Interactions** dataset containing:

| Data File | Description | Size |
|-----------|-------------|------|
| clicks_full.csv | 3M+ user-article click interactions | 220 MB |
| articles_metadata.csv | Article information (category, publisher, word count) | 11 MB |
| articles_embeddings.pickle | Pre-computed 250-dim article embedding vectors | 364 MB |

After preprocessing (filtering users with ≥ 5 interactions, articles with ≥ 3 interactions):

- **47 unique users** retained for model training
- **33 unique articles** retained for model training
- Train/test split: 80% / 20% (time-based per-user split)

---

## 3. Software Architecture

### 3.1 Architecture Components

The solution is composed of four main components:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Application** | Flask (Python) | User-facing interface for selecting users, choosing recommendation methods, and displaying results |
| **Recommendation API** | Flask REST API / Azure Functions | Serverless API that loads trained models and serves recommendations via HTTP endpoints |
| **Core ML Module** | Python (scikit-learn, scipy, numpy) | Implements CF, Content-Based, and Hybrid recommendation algorithms with training, inference, and evaluation |
| **Model Storage** | Local filesystem / Azure Blob Storage | Persists trained model artifacts (joblib serialized) |

### 3.2 Current Architecture (MVP)

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│                 (Flask Web App)                          │
│                 localhost:5002                           │
│                                                         │
│  Features:                                              │
│  - User ID selection dropdown                           │
│  - Method selection (Hybrid / CF / Content-Based)       │
│  - Display top 5 recommendations with scores            │
│  - Find similar articles                                │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP Requests
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 RECOMMENDATION API                       │
│            (Flask API / Azure Functions)                 │
│            localhost:5001 or Azure                       │
│                                                         │
│  Endpoints:                                             │
│  - GET /api/recommend/{user_id}?method=hybrid&n=5       │
│  - GET /api/users                                       │
│  - GET /api/health                                      │
│  - GET /api/similar/{article_id}                        │
│                                                         │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │  CF Model   │ │Content Model │ │  Hybrid Model    │  │
│  │  (SVD)      │ │(Embeddings)  │ │(Weighted Combo.) │  │
│  └─────────────┘ └──────────────┘ └──────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ joblib.load()
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    MODEL STORAGE                         │
│          (Local Files / Azure Blob Storage)              │
│                                                         │
│  cf_svd_model.joblib      (14 KB)                       │
│  content_based_model.joblib (150 MB)                    │
│  hybrid_model.joblib       (1 KB)                       │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Functional Description of Each Component

#### 3.3.1 Web Application (P9_03_web_app/)

- **Technology:** Flask + HTML/CSS/JavaScript
- **Port:** 5002
- **Function:** Provides a browser-based interface where users can:
  - Select a user ID from a dropdown list (loaded from the API)
  - Choose a recommendation method (Hybrid, Collaborative Filtering, Content-Based)
  - View the top N recommended articles with relevance scores
  - Search for similar articles by article ID
- **Communication:** Proxies all requests to the Recommendation API, providing a clean separation between frontend and backend.

#### 3.3.2 Recommendation API (P9_02_azure_function/)

- **Technology:** Flask REST API (local) / Azure Functions v4 (cloud)
- **Port:** 5001 (local)
- **Function:** Stateless HTTP API that:
  - Loads trained models on startup (cold start)
  - Accepts user ID as input via URL parameter
  - Returns top N article recommendations as JSON with scores
  - Supports three recommendation methods selectable via query parameter
  - Provides user listing and health check endpoints
- **Azure Deployment:** Deployed as a Python v2 programming model Azure Function on a Consumption (Serverless) plan, ensuring pay-per-execution cost efficiency.

#### 3.3.3 Core ML Module (src/)

- **Technology:** Python with scikit-learn, scipy, numpy, pandas
- **Structure:** Object-oriented design with separate classes:
  - `CollaborativeFiltering` — User-based, Item-based, and SVD matrix factorization
  - `ContentBasedRecommender` — Embedding-based with PCA reduction
  - `HybridRecommender` — Weighted combination of CF and Content-Based
  - `DataLoader` — Data loading, preprocessing, and train/test splitting
  - `utils.py` — Evaluation metrics (Precision@K, Recall@K, NDCG@K, etc.)
  - `config.py` — Centralized configuration for paths, hyperparameters, and constants

#### 3.3.4 Model Storage

- **Local:** Models saved as `.joblib` files in `P9_02_azure_function/model/`
- **Cloud:** Packaged inside the Azure Functions deployment ZIP, loaded on cold start
- **Format:** Python objects serialized with joblib for efficient numpy array storage

---

## 4. Recommendation Approaches

### 4.1 Collaborative Filtering (CF)

Collaborative Filtering leverages user-item interaction patterns without requiring content features.

#### 4.1.1 User-Based CF

- **Principle:** Find users with similar reading patterns, recommend articles they read that the target user hasn't.
- **Similarity:** Cosine similarity on the user-item interaction matrix.
- **Formula:** `score(u, i) = Σ sim(u, v) × r(v, i) / Σ |sim(u, v)|`
- **Parameters:** n_neighbors = 20, min_support = 3

#### 4.1.2 Item-Based CF

- **Principle:** Find articles similar to what the user has read, based on co-occurrence in user histories.
- **Advantage:** More stable than user-based for large catalogs since item relationships change less frequently.
- **Similarity:** Cosine similarity on the transposed user-item matrix.

#### 4.1.3 Matrix Factorization (SVD)

- **Principle:** Decompose the user-item matrix R into latent factors: R ≈ U × Σ × Vᵀ
- **Implementation:** TruncatedSVD from scikit-learn
- **Parameters:** n_factors = 10, captures hidden patterns in interaction data
- **Prediction:** Reconstruct the full matrix from latent factors to predict unseen interactions.

### 4.2 Content-Based Filtering

- **Principle:** Uses pre-computed article embeddings to compute content similarity.
- **User Profile:** Mean of embedding vectors of all articles the user has read.
- **Similarity:** Cosine similarity between user profile and candidate article embeddings.
- **PCA Reduction:** Original embeddings reduced to 250 dimensions via PCA for deployment efficiency.
- **Cold Start Handling:** For new users with no history, returns diverse articles selected via K-means clustering on embeddings.

### 4.3 Hybrid Approach

Combines CF and Content-Based methods to leverage the strengths of both.

- **Weighted Strategy:** `score = 0.6 × CF_score + 0.4 × Content_score`
- **Switching Strategy:** Use CF for active users (many interactions), Content-Based for new users.
- **Cascade Strategy:** Use CF to generate candidates, re-rank using content similarity.
- **Default:** Weighted combination (deployed model).

---

## 5. Model Evaluation

### 5.1 Evaluation Methodology

- **Split:** Time-based per-user split (80% train / 20% test). For each user, the most recent interactions form the test set.
- **Metrics:** Standard information retrieval metrics at K=5.
- **Scope:** Evaluated on all users present in both train and test sets.

### 5.2 Results

| Model | Precision@5 | Recall@5 | F1@5 | NDCG@5 | MRR | Hit Rate@5 |
|-------|-------------|----------|------|--------|-----|------------|
| User-Based CF | 0.0766 | 0.3723 | 0.1266 | 0.2811 | 0.2525 | 0.3830 |
| **Item-Based CF** | **0.0894** | **0.4255** | **0.1469** | **0.2752** | **0.2248** | **0.4255** |
| SVD | 0.0468 | 0.2340 | 0.0780 | 0.1741 | 0.1543 | 0.2340 |
| Content-Based | 0.0553 | 0.2553 | 0.0902 | 0.2008 | 0.1851 | 0.2553 |
| Hybrid | 0.0638 | 0.2979 | 0.1044 | 0.1980 | 0.1652 | 0.2979 |

### 5.3 Analysis

- **Best model:** Item-Based CF achieves the highest Precision@5 (0.0894), Recall@5 (42.55%), and Hit Rate@5 (42.55%).
- **User-Based CF** achieves the highest NDCG@5 (0.2811) and MRR (0.2525), indicating better ranking quality.
- **SVD** underperforms on this small dataset (47 users, 33 articles) — matrix factorization benefits from larger, denser datasets.
- **Content-Based** provides a solid baseline and is essential for cold-start scenarios.
- **Hybrid** blends both approaches and offers balanced performance across all metrics.
- The deployed SVD model was chosen for its API simplicity, but Item-Based CF is recommended for production with a larger dataset.

---

## 6. API Specification

### 6.1 Endpoints

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| GET | `/api/recommend/{user_id}` | Get article recommendations | `n` (default: 5), `method` (cf/content/hybrid) |
| GET | `/api/users` | List available user IDs | `limit` (default: 100) |
| GET | `/api/health` | Health check | — |
| GET | `/api/similar/{article_id}` | Find similar articles | `n` (default: 5) |

### 6.2 Example Response

```json
{
  "user_id": 42,
  "method": "hybrid",
  "recommendations": [
    {"article_id": 157541, "score": 0.8523},
    {"article_id": 68866, "score": 0.7891},
    {"article_id": 235840, "score": 0.7456},
    {"article_id": 96663, "score": 0.7234},
    {"article_id": 119592, "score": 0.6987}
  ],
  "count": 5
}
```

---

## 7. Target Architecture for New Users and Articles

### 7.1 Production Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                       USERS                              │
│            (Web Browser / Mobile App)                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              AZURE CDN / LOAD BALANCER                   │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌──────────────────┐       ┌──────────────────────────────┐
│  Web Application │       │   Azure Functions (API)      │
│ (Azure App       │       │  (Consumption/Serverless)    │
│  Service)        │       │  - /recommend/{user_id}      │
│                  │       │  - /users                    │
│                  │       │  - /health                   │
└──────────────────┘       └──────────────┬───────────────┘
                                          │
                  ┌───────────────────────┼──────────────┐
                  ▼                       ▼              ▼
        ┌──────────────────┐   ┌──────────────┐  ┌─────────────┐
        │  Azure Blob      │   │ Azure Cosmos │  │ Azure ML    │
        │  Storage         │   │ DB           │  │ (Model      │
        │  (Models)        │   │ (User Data,  │  │  Retraining)│
        │                  │   │  Articles)   │  │             │
        └──────────────────┘   └──────┬───────┘  └─────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  Azure Event Hub /     │
                          │  Service Bus Queue     │
                          │  (New User/Article     │
                          │   Events)              │
                          └────────────┬───────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │  Azure Functions        │
                          │  (Batch Model           │
                          │   Retraining Trigger)   │
                          └────────────────────────┘
```

### 7.2 Adding New Users

| Step | Component | Action |
|------|-----------|--------|
| 1 | Web App | New user signs up, profile stored in Cosmos DB |
| 2 | Event Hub | "New User" event published |
| 3 | API | Cold-start: Content-Based model provides initial recommendations based on user preferences or popular articles |
| 4 | Cosmos DB | As user clicks articles, interactions are logged |
| 5 | Retraining Function | Periodic batch job (e.g., nightly) retrains CF model with new interaction data |
| 6 | API | After retraining, CF and Hybrid models can generate personalized recommendations |

### 7.3 Adding New Articles

| Step | Component | Action |
|------|-----------|--------|
| 1 | CMS / Publisher | New article created and published |
| 2 | Azure ML | NLP model generates embedding vector for the new article |
| 3 | Blob Storage | Embedding stored, Content-Based index updated immediately |
| 4 | API | New article is immediately available for Content-Based recommendations |
| 5 | Cosmos DB | As users interact with the article, click data is logged |
| 6 | Retraining Function | Next batch retraining incorporates the new article into CF model |

### 7.4 Key Design Decisions for Scalability

- **Serverless (Consumption plan):** Auto-scales to zero, pay-per-request, handles bursty traffic.
- **Event-driven retraining:** Models are retrained on a schedule (not per-request), balancing freshness with cost.
- **Content-Based cold-start:** Ensures new users and new articles receive recommendations immediately, without waiting for retraining.
- **Cosmos DB:** Globally distributed, low-latency document store for user profiles and interaction logs.
- **Separation of concerns:** Web App, API, and Model Training are independent components that can be scaled, updated, and deployed independently.

---

## 8. Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Frontend | Flask + HTML/CSS/JS | Lightweight, sufficient for MVP |
| API | Flask / Azure Functions v4 | Serverless, cost-efficient, auto-scaling |
| ML | scikit-learn, scipy, numpy | Mature, well-documented, production-ready |
| Data | pandas | Efficient data manipulation |
| Serialization | joblib | Optimized for numpy arrays and sklearn models |
| Cloud | Azure (Functions, Blob, Cosmos DB) | Meets project requirements |
| Version Control | GitHub | Code versioning and collaboration |

---

## 9. Conclusion

The "My Content" recommendation MVP successfully implements three recommendation approaches (Collaborative Filtering, Content-Based, and Hybrid) with a clean architecture separating the web interface, API, and ML models. The system meets the business requirement of recommending 5 articles per user and is designed to scale through serverless Azure Functions.

The target architecture accommodates new users (via Content-Based cold-start) and new articles (via real-time embedding generation), with event-driven batch retraining to continuously improve recommendations. Item-Based CF showed the strongest evaluation results, and the Hybrid approach provides a good balance of accuracy and cold-start coverage.

---

## Appendix A: Repository Structure

```
Project9/
├── src/                           # Core recommendation module
│   ├── __init__.py
│   ├── config.py                  # Configuration & constants
│   ├── data_loader.py             # Data loading & preprocessing
│   ├── collaborative_filtering.py # CF models (User/Item/SVD)
│   ├── content_based.py           # Content-based recommender
│   ├── hybrid_recommender.py      # Hybrid approach
│   └── utils.py                   # Evaluation metrics & visualization
├── P9_01_scripts/
│   └── P9_01_notebook.ipynb       # Training & evaluation notebook
├── P9_02_azure_function/
│   ├── app.py                     # Flask REST API
│   ├── function_app.py            # Azure Function implementation
│   ├── host.json                  # Azure Function config
│   ├── requirements.txt           # API dependencies
│   └── model/                     # Trained model artifacts
├── P9_03_web_app/
│   ├── app.py                     # Flask web application
│   ├── templates/index.html       # Main page template
│   └── static/style.css           # Application styles
├── deploy_to_azure.sh             # Azure deployment script
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Appendix B: GitHub Repository

**URL:** https://github.com/saidurga24/content-recommendation

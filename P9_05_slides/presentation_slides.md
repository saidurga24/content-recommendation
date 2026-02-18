# Presentation Slides — Project 9: Content Recommendation System

**Instructions for Copilot:** Generate a professional PowerPoint presentation from this content. Use a modern dark/tech theme. Each "## Slide" heading is a separate slide. Keep text concise on slides — use bullet points, tables, and diagrams.

---

## Slide 1 — Title Slide

**My Content — Article Recommendation System**

Project 9: Create an Application for Recommending Content

Sai Durga Prasad  
AI Engineer — OpenClassrooms  
February 2026

---

## Slide 2 — Agenda

1. Modeling Approaches Tested (10 min)
2. Recommendation System Features in the Application (6 min)
3. Target Technical Architecture (2 min)
4. Live Application Demo (2 min)

---

## Slide 3 — Business Context

**"My Content"** — A startup that encourages reading by recommending relevant articles

**User Story:**
> "As a user, I receive a selection of five recommended articles."

**Goal:** Build an MVP recommendation system that:
- Recommends 5 relevant articles per user
- Uses serverless architecture (Azure Functions)
- Handles new users and new articles

---

## Slide 4 — Dataset Overview

**Globo.com News Portal User Interactions**

| Data | Description | Size |
|------|-------------|------|
| User Clicks | 3M+ user-article click interactions | 220 MB |
| Article Metadata | Category, publisher, word count | 11 MB |
| Article Embeddings | Pre-computed 250-dim vectors | 364 MB |

**After preprocessing:**
- 47 users (≥5 interactions each)
- 33 articles (≥3 interactions each)
- Train/test split: 80%/20% (time-based)

---

## Slide 5 — Approach 1: Collaborative Filtering

**Principle:** Recommend based on interaction patterns, no content features needed

Three variants tested:

| Method | How It Works |
|--------|-------------|
| **User-Based CF** | Find similar users → recommend what they read |
| **Item-Based CF** | Find articles similar to what user read (co-occurrence) |
| **SVD (Matrix Factorization)** | Decompose user-item matrix into latent factors: R ≈ U × Σ × Vᵀ |

**Similarity:** Cosine similarity  
**SVD Parameters:** n_factors = 10

---

## Slide 6 — Approach 2: Content-Based Filtering

**Principle:** Use article embeddings to match content to user preferences

**How it works:**
1. Each article has a 250-dim embedding vector (PCA-reduced)
2. User profile = mean of embeddings of all articles they've read
3. Score = cosine similarity between user profile and candidate articles

**Strengths:**
- ✅ Handles cold-start for new articles
- ✅ No need for interaction history from other users
- ✅ Immediate recommendations possible

---

## Slide 7 — Approach 3: Hybrid Method

**Principle:** Combine CF and Content-Based to leverage both strengths

**Strategies implemented:**

| Strategy | Description |
|----------|-------------|
| **Weighted** | `score = 0.6 × CF + 0.4 × Content` |
| **Switching** | CF for active users, Content-Based for new users |
| **Cascade** | CF generates candidates, Content re-ranks |

**Deployed model:** Weighted combination (default)

---

## Slide 8 — Evaluation Metrics

**Standard information retrieval metrics at K=5:**

| Metric | What it measures |
|--------|-----------------|
| **Precision@5** | Fraction of recommendations that are relevant |
| **Recall@5** | Fraction of relevant items found in top 5 |
| **F1@5** | Harmonic mean of Precision and Recall |
| **NDCG@5** | Ranking quality (position-aware) |
| **MRR** | Position of first relevant recommendation |
| **Hit Rate@5** | Did at least 1 relevant article appear in top 5? |

---

## Slide 9 — Evaluation Results

| Model | Precision@5 | Recall@5 | F1@5 | NDCG@5 | MRR | Hit Rate@5 |
|-------|-------------|----------|------|--------|-----|------------|
| User-Based CF | 0.077 | 0.372 | 0.127 | **0.281** | **0.253** | 0.383 |
| **Item-Based CF** | **0.089** | **0.426** | **0.147** | 0.275 | 0.225 | **0.426** |
| SVD | 0.047 | 0.234 | 0.078 | 0.174 | 0.154 | 0.234 |
| Content-Based | 0.055 | 0.255 | 0.090 | 0.201 | 0.185 | 0.255 |
| Hybrid | 0.064 | 0.298 | 0.104 | 0.198 | 0.165 | 0.298 |

**Best overall: Item-Based CF** (highest Precision, Recall, Hit Rate)  
**Best ranking quality: User-Based CF** (highest NDCG, MRR)

---

## Slide 10 — Model Comparison Chart

*(Include the bar chart from P9_01_scripts/model_comparison.png)*

**Key Insights:**
- Item-Based CF outperforms all other models on most metrics
- SVD underperforms on this small dataset (47 users) — better suited for larger datasets
- Hybrid provides balanced performance and cold-start coverage
- Content-Based is essential for new users/articles with no interaction history

---

## Slide 11 — Current Architecture (MVP)

```
┌───────────────────────────────────────────┐
│          USER INTERFACE                    │
│        (Flask Web App)                     │
│        Port 5002                           │
└──────────────────┬────────────────────────┘
                   │ HTTP
                   ▼
┌───────────────────────────────────────────┐
│       RECOMMENDATION API                   │
│    (Flask / Azure Functions)               │
│    Port 5001 or Azure Cloud                │
│                                            │
│  ┌──────┐  ┌─────────┐  ┌──────────────┐  │
│  │  CF  │  │Content  │  │   Hybrid     │  │
│  │Model │  │Model    │  │   Model      │  │
│  └──────┘  └─────────┘  └──────────────┘  │
└──────────────────┬────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────┐
│          MODEL STORAGE                     │
│    (Local Files / Azure Blob)              │
└───────────────────────────────────────────┘
```

---

## Slide 12 — Application Features

The Flask web application provides:

1. **User Selection** — Dropdown of available user IDs
2. **Method Selection** — Choose Hybrid, CF, or Content-Based
3. **Recommendations Display** — Top 5 articles with relevance scores
4. **Similar Articles** — Search by article ID
5. **API Status** — Real-time connection status indicator

**API Endpoints (Live on Azure):**
- `GET https://func-mycontent-recommendation.azurewebsites.net/api/recommend/{user_id}?method=hybrid&n=5`
- `GET https://func-mycontent-recommendation.azurewebsites.net/api/users`
- `GET https://func-mycontent-recommendation.azurewebsites.net/api/health`
- `GET https://func-mycontent-recommendation.azurewebsites.net/api/similar/{article_id}`

---

## Slide 13 — API Response Example

```json
{
  "user_id": 42,
  "method": "hybrid",
  "recommendations": [
    {"article_id": 157541, "score": 0.8523},
    {"article_id": 68866,  "score": 0.7891},
    {"article_id": 235840, "score": 0.7456},
    {"article_id": 96663,  "score": 0.7234},
    {"article_id": 119592, "score": 0.6987}
  ],
  "count": 5
}
```

Input: User ID → Output: Top 5 recommended articles with scores

---

## Slide 14 — Target Architecture (Production)

```
         ┌──────────┐
         │  Users   │
         │(Web/Mobile)│
         └────┬─────┘
              │
         ┌────▼─────┐
         │Azure CDN /│
         │Load Bal.  │
         └────┬─────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌────────┐    ┌──────────────┐
│Web App │    │Azure Functions│
│(App    │    │(Serverless    │
│Service)│    │ API)          │
└────────┘    └──────┬───────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌────────┐
    │Blob     │ │Cosmos   │ │Azure   │
    │Storage  │ │DB       │ │ML      │
    │(Models) │ │(Users,  │ │(Retrain│
    │         │ │Articles)│ │ing)    │
    └─────────┘ └────┬────┘ └────────┘
                     │
              ┌──────▼───────┐
              │Event Hub /   │
              │Service Bus   │
              │(New User/    │
              │Article Events)│
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │Azure Function│
              │(Batch Model  │
              │ Retraining)  │
              └──────────────┘
```

---

## Slide 15 — Handling New Users

| Step | What Happens |
|------|-------------|
| 1 | New user signs up → profile stored in Cosmos DB |
| 2 | "New User" event sent to Event Hub |
| 3 | **Cold-start:** Content-Based model provides initial recommendations (popular articles / diverse selection) |
| 4 | As user clicks articles, interactions are logged |
| 5 | Nightly batch retraining updates CF model |
| 6 | After retraining, personalized CF + Hybrid recommendations available |

**Key:** Content-Based ensures immediate recommendations even with no history

---

## Slide 16 — Handling New Articles

| Step | What Happens |
|------|-------------|
| 1 | New article published |
| 2 | NLP model generates 250-dim embedding vector |
| 3 | Embedding stored → Content-Based index updated **immediately** |
| 4 | Article instantly available for Content-Based recommendations |
| 5 | As users interact, click data is logged |
| 6 | Next batch retraining incorporates article into CF model |

**Key:** Embedding-based indexing enables instant discoverability

---

## Slide 17 — Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Flask + HTML/CSS/JavaScript |
| API | Flask / Azure Functions v4 (Python) |
| ML | scikit-learn, scipy, numpy, pandas |
| Serialization | joblib |
| Cloud | Azure (Functions, Blob Storage, Cosmos DB) |
| Version Control | GitHub |

**Repository:** https://github.com/saidurga24/content-recommendation

---

## Slide 18 — Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Serverless (Consumption plan)** | Auto-scales to zero, pay-per-request |
| **Event-driven retraining** | Balances model freshness with cost |
| **Content-Based cold-start** | Immediate recommendations for new users/articles |
| **Modular OOP design** | Each model class is independently trainable, testable, deployable |
| **Separate Web App + API** | Clean separation of concerns, independent scaling |

---

## Slide 19 — Demo

**Live demonstration of the application:**

1. Open Azure Function health check: `https://func-mycontent-recommendation.azurewebsites.net/api/health`
2. Show users list: `https://func-mycontent-recommendation.azurewebsites.net/api/users`
3. Get recommendations: `https://func-mycontent-recommendation.azurewebsites.net/api/recommend/102?method=hybrid&n=5`
4. Start the Flask web app locally (connected to Azure API)
5. Select a user from the dropdown
6. View top 5 recommended articles with scores

---

## Slide 20 — Summary & Next Steps

**What was delivered:**
- ✅ Three recommendation approaches (CF, Content-Based, Hybrid)
- ✅ Serverless Azure Function API
- ✅ Flask web application
- ✅ Comprehensive evaluation with 6 metrics
- ✅ Architecture for new users and articles

**Potential improvements:**
- Larger dataset for better SVD performance
- Deep learning embeddings (BERT) for richer article features
- A/B testing framework for comparing methods in production
- Real-time model updates (online learning)

**Thank you!**

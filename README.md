# Project 9: Content Recommendation System

## My Content - Article Recommendation MVP

**OpenClassrooms - AI Engineer Path**
**Author:** Sai Durga Prasad

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Context](#business-context)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Recommendation Approaches](#recommendation-approaches)
8. [API Documentation](#api-documentation)
9. [Web Application](#web-application)
10. [Azure Functions Deployment](#azure-functions-deployment)
11. [Architecture](#architecture)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Deliverables](#deliverables)

---

## Project Overview

This project implements an article recommendation system for **"My Content"**, a startup that wants to encourage reading by recommending relevant content to its users. The MVP delivers personalized recommendations of 5 articles to each user based on their interaction history.

### Key Features

- **Collaborative Filtering**: User-based, Item-based, and SVD matrix factorization
- **Content-Based Filtering**: Using pre-computed article embeddings
- **Hybrid Approach**: Combines CF and content-based methods
- **REST API**: Flask-based API for serving recommendations
- **Azure Functions**: Serverless deployment option
- **Web Interface**: Flask application for user interaction

---

## Business Context

**Scenario**: You are the CTO and co-founder of "My Content" startup, working with Samia (CEO) to develop an MVP for recommending articles and books to users.

**User Story**:
> "As a user of the application, I will receive a selection of five articles"

**Requirements**:
- Recommend 5 relevant articles per user
- Support adding new users and articles
- Use serverless architecture (Azure Functions)
- Implement both CF and content-based approaches

---

## Dataset

### Globo.com News Portal User Interactions

The dataset contains user interactions with articles from a Brazilian news portal.

**Location**: `data/news-portal-user-interactions-by-globocom/`

### Files

| File | Size | Description |
|------|------|-------------|
| `clicks_sample.csv` | 134 KB | Sample of user-article interactions |
| `clicks.zip` | 38 MB | Full click data (compressed) |
| `articles_metadata.csv` | 11 MB | Article information |
| `articles_embeddings.pickle` | 364 MB | Pre-computed article embeddings |

### Data Schema

**clicks_sample.csv**
| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Unique user identifier |
| session_id | int | Session identifier |
| session_start | timestamp | Session start time |
| session_size | int | Number of clicks in session |
| click_article_id | int | Article that was clicked |
| click_timestamp | timestamp | Time of click |
| click_environment | int | Environment code |
| click_deviceGroup | int | Device type |
| click_os | int | Operating system |
| click_country | int | Country code |
| click_region | int | Region code |
| click_referrer_type | int | Referrer type |

**articles_metadata.csv**
| Column | Type | Description |
|--------|------|-------------|
| article_id | int | Unique article identifier |
| category_id | int | Article category |
| created_at_ts | timestamp | Publication timestamp |
| publisher_id | int | Publisher identifier |
| words_count | int | Number of words in article |

**articles_embeddings.pickle**
- Dictionary mapping article_id to embedding vector
- Embedding dimension: 250 (original may vary, reduced via PCA)

---

## Project Structure

```
Project9/
│
├── src/                                    # Core recommendation module
│   ├── __init__.py                         # Package initialization
│   ├── config.py                           # Configuration & constants
│   ├── data_loader.py                      # Data loading & preprocessing
│   ├── collaborative_filtering.py          # CF models (User/Item/SVD)
│   ├── content_based.py                    # Content-based recommender
│   ├── hybrid_recommender.py               # Hybrid approach
│   └── utils.py                            # Evaluation metrics & visualization
│
├── data/
│   └── news-portal-user-interactions-by-globocom/
│       ├── clicks_sample.csv               # User interactions
│       ├── clicks.zip                      # Full click data
│       ├── articles_metadata.csv           # Article info
│       └── articles_embeddings.pickle      # Article embeddings
│
├── P9_01_scripts/
│   └── P9_01_notebook.ipynb                # Training notebook
│
├── P9_02_azure_function/                   # API & Azure Function
│   ├── app.py                              # Flask REST API
│   ├── function_app.py                     # Azure Function implementation
│   ├── host.json                           # Azure Function config
│   ├── local.settings.json.example         # Local settings template
│   ├── requirements.txt                    # API dependencies
│   ├── Procfile                            # Deployment config
│   └── model/                              # Trained models (generated)
│       ├── cf_svd_model.joblib
│       ├── content_based_model.joblib
│       └── hybrid_model.joblib
│
├── P9_03_web_app/                          # Flask web interface
│   ├── app.py                              # Flask web application
│   ├── static/
│   │   └── style.css                       # Application styles
│   ├── templates/
│   │   └── index.html                      # Main page template
│   └── requirements.txt                    # Web app dependencies
│
├── P9_04_technical_note/                   # Technical documentation
│   └── (technical_note.pdf)
│
├── P9_05_slides/                           # Presentation slides
│   └── (presentation.pptx)
│
├── scripts/
│   └── download_data.py                    # Data download helper
│
├── requirements.txt                        # All project dependencies
├── .gitignore                              # Git ignore rules
└── README.md                               # This file
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for version control)
- Azure account (for deployment, optional)

### Step 1: Clone/Navigate to Project

```bash
cd /Users/durga/OpenClassrooms/Project9
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Data

Ensure the data files are in the correct location:
```bash
ls data/news-portal-user-interactions-by-globocom/
```

Expected output:
```
articles_embeddings.pickle
articles_metadata.csv
clicks_sample.csv
clicks.zip
```

---

## Usage Guide

### 1. Training Models (Jupyter Notebook)

```bash
# Activate virtual environment
source venv/bin/activate

# Start Jupyter
jupyter notebook P9_01_scripts/P9_01_notebook.ipynb
```

The notebook will:
- Load and explore the data
- Preprocess interactions
- Train Collaborative Filtering models (User-based, Item-based, SVD)
- Train Content-Based model
- Create Hybrid model
- Evaluate all models
- Save trained models to `P9_02_azure_function/model/`

### 2. Start the Flask API

```bash
# Activate virtual environment
source venv/bin/activate

# Navigate to API folder
cd P9_02_azure_function

# Start the API
python app.py
```

API will be available at: `http://localhost:5001`

### 3. Start the Web Application

```bash
# In a new terminal, activate virtual environment
source venv/bin/activate

# Navigate to web app folder
cd P9_03_web_app

# Start Flask web app
python app.py
```

Web app will be available at: `http://localhost:5002`

---

## Recommendation Approaches

### 1. Collaborative Filtering (CF)

Based on user-item interactions without using item content.

#### User-Based CF
- Finds users with similar interaction patterns
- Recommends articles that similar users liked
- Formula: `score(u, i) = Σ sim(u, v) × r(v, i) / Σ |sim(u, v)|`

#### Item-Based CF
- Finds articles similar to what user has read
- Based on co-occurrence in user histories
- More stable than user-based for large catalogs

#### Matrix Factorization (SVD)
- Decomposes user-item matrix into latent factors
- Captures hidden patterns in interactions
- `R ≈ U × Σ × V^T`
- Parameters: 50 latent factors

### 2. Content-Based Filtering

Uses article embeddings to find similar content.

- Pre-computed 250-dimensional embeddings (PCA reduced)
- User profile = mean of read article embeddings
- Cosine similarity for matching
- Handles cold-start for new articles

### 3. Hybrid Approach

Combines CF and Content-Based methods.

**Strategies:**
- **Weighted**: `score = 0.6 × CF_score + 0.4 × Content_score`
- **Switching**: Use CF for active users, Content for new users
- **Cascade**: Use CF candidates, re-rank with content similarity

---

## API Documentation

### Base URL
```
http://localhost:5001
```

### Endpoints

#### GET `/`
Returns API status and available endpoints.

**Response:**
```json
{
  "status": "running",
  "name": "Article Recommendation API",
  "version": "1.0.0",
  "endpoints": {...},
  "models_loaded": {
    "collaborative_filtering": true,
    "content_based": true,
    "hybrid": true
  }
}
```

#### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

#### GET `/users`
List available user IDs.

**Parameters:**
- `limit` (optional): Max users to return (default: 100)

**Response:**
```json
{
  "users": [0, 1, 2, 3, ...],
  "total": 1000,
  "returned": 100
}
```

#### GET `/articles`
List available article IDs.

**Parameters:**
- `limit` (optional): Max articles to return (default: 100)

**Response:**
```json
{
  "articles": [157541, 68866, 235840, ...],
  "total": 5000,
  "returned": 100
}
```

#### GET `/recommend/<user_id>`
Get article recommendations for a user.

**Parameters:**
- `n` (optional): Number of recommendations (default: 5)
- `method` (optional): `cf`, `content`, or `hybrid` (default: hybrid)

**Example:**
```
GET /recommend/42?n=5&method=hybrid
```

**Response:**
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

#### GET `/similar/<article_id>`
Get articles similar to a given article.

**Parameters:**
- `n` (optional): Number of similar articles (default: 5)

**Response:**
```json
{
  "article_id": 157541,
  "similar_articles": [
    {"article_id": 68866, "similarity": 0.9234},
    ...
  ],
  "count": 5
}
```

---

## Web Application

The Flask web application provides a user-friendly interface for:

1. **User Selection**: Choose from available user IDs
2. **Method Selection**: Pick recommendation approach (Hybrid, CF, Content-Based)
3. **Get Recommendations**: Display top 5 recommended articles with scores
4. **Similar Articles**: Find articles similar to a given article ID

### Features

- Real-time API status indicator
- Adjustable number of recommendations (3, 5, 7, 10)
- Score visualization with color-coded bars
- Animated recommendation cards
- Modern dark theme with responsive design

---

## Azure Functions Deployment

### Option 1: Azure Portal Deployment

1. **Create Function App**
   - Go to Azure Portal → Create Resource → Function App
   - Select: Consumption (Serverless) plan
   - Runtime: Python 3.9+
   - Region: Choose nearest

2. **Deploy Code**
   - Use VS Code Azure Functions extension
   - Or use Azure CLI:
   ```bash
   func azure functionapp publish <function-app-name>
   ```

3. **Upload Models to Blob Storage**
   - Create Azure Blob Storage account
   - Upload model files from `P9_02_azure_function/model/`
   - Update function to load from Blob Storage

### Option 2: Local Azure Functions

```bash
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Navigate to function folder
cd P9_02_azure_function

# Create local.settings.json from example
cp local.settings.json.example local.settings.json

# Start locally
func start
```

### Azure Function Endpoints

```
https://<function-app-name>.azurewebsites.net/api/recommend/{user_id}
https://<function-app-name>.azurewebsites.net/api/users
https://<function-app-name>.azurewebsites.net/api/health
```

---

## Architecture

### Current Architecture (MVP)

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                     (Flask Web App)                              │
│                     localhost:5002                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP Requests
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RECOMMENDATION API                          │
│                   (Flask / Azure Functions)                      │
│                     localhost:5001                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   CF Model  │  │Content Model│  │    Hybrid Model         │  │
│  │   (SVD)     │  │(Embeddings) │  │ (Weighted Combination)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Load Models
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL STORAGE                               │
│              (Local Files / Azure Blob Storage)                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │cf_svd_model.pkl │  │content_model.pkl│  │hybrid_model.pkl │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Target Architecture (Production)

```
┌─────────────────────────────────────────────────────────────────┐
│                         USERS                                    │
│              (Web Browser / Mobile App)                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AZURE CDN / LOAD BALANCER                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────────────────┐
│   Web Application   │       │     Azure Functions (API)       │
│  (Azure App Service)│       │   (Consumption/Serverless)      │
└─────────────────────┘       └─────────────────┬───────────────┘
                                                │
                    ┌───────────────────────────┼───────────────┐
                    ▼                           ▼               ▼
          ┌─────────────────┐       ┌─────────────────┐ ┌──────────────┐
          │  Azure Blob     │       │  Azure Cosmos   │ │Azure ML      │
          │  Storage        │       │  DB             │ │(Model Train) │
          │  (Models)       │       │  (User Data)    │ │              │
          └─────────────────┘       └─────────────────┘ └──────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │   Event Hub / Queue     │
                              │ (New User/Article Events)│
                              └─────────────────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │   Azure Functions       │
                              │ (Model Retraining)      │
                              └─────────────────────────┘
```

### Adding New Users & Articles

**New User Flow:**
1. User signs up → Event sent to Event Hub
2. Initial recommendations via Content-Based (cold start)
3. As user interacts, CF model learns preferences
4. Periodic batch retraining updates models

**New Article Flow:**
1. Article published → Event sent to Event Hub
2. Generate embedding via NLP model
3. Add to content index immediately
4. CF model learns from interactions over time

---

## Evaluation Metrics

### Metrics Used

| Metric | Description | Formula |
|--------|-------------|---------|
| Precision@K | Fraction of recommended items that are relevant | `relevant ∩ recommended / K` |
| Recall@K | Fraction of relevant items that are recommended | `relevant ∩ recommended / total_relevant` |
| F1@K | Harmonic mean of Precision and Recall | `2 × P × R / (P + R)` |
| NDCG@K | Normalized Discounted Cumulative Gain | Considers ranking position |
| Hit Rate@K | 1 if any relevant item in top K | Binary metric |
| MRR | Mean Reciprocal Rank | `1 / position_of_first_hit` |

### Expected Results

| Model | Precision@5 | Recall@5 | NDCG@5 | Hit Rate@5 |
|-------|-------------|----------|--------|------------|
| User-Based CF | ~0.15 | ~0.08 | ~0.12 | ~0.35 |
| Item-Based CF | ~0.18 | ~0.10 | ~0.15 | ~0.40 |
| SVD | ~0.20 | ~0.11 | ~0.17 | ~0.45 |
| Content-Based | ~0.12 | ~0.07 | ~0.10 | ~0.30 |
| **Hybrid** | **~0.22** | **~0.12** | **~0.19** | **~0.48** |

*Note: Actual results depend on dataset characteristics and hyperparameters.*

---

## Deliverables

### P9_01_scripts
- Jupyter notebook with EDA, model training, and evaluation
- Generated visualizations (PNG files)

### P9_02_azure_function
- Flask API (`app.py`)
- Azure Function implementation (`function_app.py`)
- Trained models in `model/` folder

### P9_03_web_app
- Flask application demonstrating recommendations

### P9_04_technical_note
- Technical documentation (PDF)
- Architecture diagrams
- Functional descriptions

### P9_05_slides
- Presentation slides covering:
  - Modeling approaches (10 min)
  - Recommendation system features (6 min)
  - Target architecture (2 min)
  - Application demo (2 min)

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**2. API Connection Failed**
```bash
# Check if API is running
curl http://localhost:5001/health

# Start API if not running
cd P9_02_azure_function && python app.py
```

**3. Models Not Found**
```bash
# Run the training notebook first to generate models
jupyter notebook P9_01_scripts/P9_01_notebook.ipynb
```

**4. Memory Error with Embeddings**
```python
# Use PCA to reduce embedding size in config.py
CONTENT_PARAMS = {
    'embedding_dim': 100,  # Reduce from 250
    'use_pca': True
}
```

---

## License

This project is for educational purposes as part of the OpenClassrooms AI Engineer program.

---

## Contact

**Student:** Sai Durga Prasad
**Program:** AI Engineer - OpenClassrooms
**Project:** P9 - Create an Application for Recommending Content

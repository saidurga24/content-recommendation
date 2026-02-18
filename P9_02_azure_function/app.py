"""
Flask API for Article Recommendation System
Provides endpoints to get recommendations for users

This API can be deployed as:
1. Azure Functions (serverless)
2. Standalone Flask API
3. AWS Lambda
"""

import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import API_HOST, API_PORT, MODEL_DIR, N_RECOMMENDATIONS
from src.collaborative_filtering import CollaborativeFiltering
from src.content_based import ContentBasedRecommender
from src.hybrid_recommender import HybridRecommender

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model instances (lazy loaded)
cf_model = None
content_model = None
hybrid_model = None
data_loader = None


def load_models():
    """Load all trained models."""
    global cf_model, content_model, hybrid_model, data_loader

    cf_path = os.path.join(MODEL_DIR, 'cf_svd_model.joblib')
    content_path = os.path.join(MODEL_DIR, 'content_based_model.joblib')
    hybrid_path = os.path.join(MODEL_DIR, 'hybrid_model.joblib')

    try:
        if os.path.exists(cf_path):
            cf_model = CollaborativeFiltering.load(cf_path)
            print("Loaded CF model")

        if os.path.exists(content_path):
            content_model = ContentBasedRecommender.load(content_path)
            print("Loaded Content-Based model")

        if os.path.exists(hybrid_path) and cf_model and content_model:
            hybrid_model = HybridRecommender.load(hybrid_path, cf_model, content_model)
            print("Loaded Hybrid model")

    except Exception as e:
        print(f"Error loading models: {e}")


# Load models on startup
load_models()


@app.route('/')
def index():
    """API status and information."""
    return jsonify({
        'status': 'running',
        'name': 'Article Recommendation API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'API status and info',
            '/users': 'List available user IDs',
            '/articles': 'List available articles',
            '/recommend/<user_id>': 'Get recommendations for user',
            '/recommend/<user_id>/<method>': 'Get recommendations using specific method',
            '/similar/<article_id>': 'Get similar articles',
            '/health': 'Health check endpoint'
        },
        'models_loaded': {
            'collaborative_filtering': cf_model is not None,
            'content_based': content_model is not None,
            'hybrid': hybrid_model is not None
        }
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': cf_model is not None or content_model is not None
    })


@app.route('/users')
def get_users():
    """Get list of available user IDs."""
    if cf_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        users = [int(uid) for uid in cf_model.user_id_map.keys()]

        # Limit to first 100 for API response
        limit = request.args.get('limit', 100, type=int)
        users = users[:limit]

        return jsonify({
            'users': users,
            'total': len(cf_model.user_id_map),
            'returned': len(users)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/articles')
def get_articles():
    """Get list of available articles."""
    if cf_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        articles = [int(aid) for aid in cf_model.article_id_map.keys()]

        # Limit to first 100 for API response
        limit = request.args.get('limit', 100, type=int)
        articles = articles[:limit]

        return jsonify({
            'articles': articles,
            'total': len(cf_model.article_id_map),
            'returned': len(articles)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/recommend/<int:user_id>')
def recommend(user_id):
    """
    Get article recommendations for a user.

    Query params:
        n: Number of recommendations (default: 5)
        method: Recommendation method (cf, content, hybrid)
    """
    n = request.args.get('n', N_RECOMMENDATIONS, type=int)
    method = request.args.get('method', 'hybrid').lower()

    try:
        if method == 'cf' and cf_model:
            model = cf_model
        elif method == 'content' and content_model:
            model = content_model
        elif method == 'hybrid' and hybrid_model:
            model = hybrid_model
        elif cf_model:
            model = cf_model
        elif content_model:
            model = content_model
        else:
            return jsonify({'error': 'No model available'}), 500

        # Get recommendations
        recommendations = model.recommend(user_id, n_recommendations=n)

        # Format response
        result = {
            'user_id': user_id,
            'method': method,
            'recommendations': [
                {'article_id': int(aid), 'score': float(score)}
                for aid, score in recommendations
            ],
            'count': len(recommendations)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'user_id': user_id
        }), 500


@app.route('/recommend/<int:user_id>/<method>')
def recommend_with_method(user_id, method):
    """Get recommendations using a specific method."""
    n = request.args.get('n', N_RECOMMENDATIONS, type=int)

    try:
        if method == 'cf':
            if cf_model is None:
                return jsonify({'error': 'CF model not loaded'}), 500
            model = cf_model
        elif method == 'content':
            if content_model is None:
                return jsonify({'error': 'Content model not loaded'}), 500
            model = content_model
        elif method == 'hybrid':
            if hybrid_model is None:
                return jsonify({'error': 'Hybrid model not loaded'}), 500
            model = hybrid_model
        else:
            return jsonify({'error': f'Unknown method: {method}'}), 400

        recommendations = model.recommend(user_id, n_recommendations=n)

        return jsonify({
            'user_id': user_id,
            'method': method,
            'recommendations': [
                {'article_id': int(aid), 'score': float(score)}
                for aid, score in recommendations
            ],
            'count': len(recommendations)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/similar/<int:article_id>')
def similar_articles(article_id):
    """Get similar articles to a given article."""
    n = request.args.get('n', N_RECOMMENDATIONS, type=int)

    try:
        if content_model:
            similar = content_model.recommend_similar_articles(article_id, n=n)
        elif cf_model and hasattr(cf_model, 'get_similar_items'):
            similar = cf_model.get_similar_items(article_id, n=n)
        else:
            return jsonify({'error': 'No model available for similarity'}), 500

        return jsonify({
            'article_id': article_id,
            'similar_articles': [
                {'article_id': int(aid), 'similarity': float(sim)}
                for aid, sim in similar
            ],
            'count': len(similar)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/similar_users/<int:user_id>')
def similar_users(user_id):
    """Get similar users to a given user (requires user-based CF model)."""
    n = request.args.get('n', 10, type=int)

    try:
        if cf_model is None:
            return jsonify({'error': 'CF model not loaded'}), 500

        if cf_model.user_similarity is None:
            return jsonify({
                'error': 'Similar users requires a user-based CF model. '
                         'Current model is SVD-based.'
            }), 400

        similar = cf_model.get_similar_users(user_id, n=n)

        return jsonify({
            'user_id': user_id,
            'similar_users': [
                {'user_id': int(uid), 'similarity': float(sim)}
                for uid, sim in similar
            ],
            'count': len(similar)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reload_models', methods=['POST'])
def reload_models():
    """Reload all models (admin endpoint)."""
    try:
        load_models()
        return jsonify({
            'status': 'success',
            'models_loaded': {
                'collaborative_filtering': cf_model is not None,
                'content_based': content_model is not None,
                'hybrid': hybrid_model is not None
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print(f"Starting API server on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=True)

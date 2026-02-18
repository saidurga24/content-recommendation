"""
Azure Functions implementation for Article Recommendation System
HTTP Trigger function that returns article recommendations

To deploy:
1. Create Azure Function App (Consumption plan)
2. Deploy this code using VS Code Azure Functions extension or CLI
3. Upload trained models to Azure Blob Storage
"""

import azure.functions as func
import json
import logging
import os
import sys

# Add parent directory and current directory to path for imports
# (parent for local dev, current dir for Azure deployment where src/ is copied alongside)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Global model instances
cf_model = None
content_model = None
hybrid_model = None


def load_models():
    """Load models - called on cold start."""
    global cf_model, content_model, hybrid_model

    from src.collaborative_filtering import CollaborativeFiltering
    from src.content_based import ContentBasedRecommender
    from src.hybrid_recommender import HybridRecommender

    # Use local model/ directory (works both locally and in Azure deployment)
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

    try:
        cf_path = os.path.join(model_dir, 'cf_svd_model.joblib')
        content_path = os.path.join(model_dir, 'content_based_model.joblib')

        if os.path.exists(cf_path):
            cf_model = CollaborativeFiltering.load(cf_path)
            logging.info("Loaded CF model")

        if os.path.exists(content_path):
            content_model = ContentBasedRecommender.load(content_path)
            logging.info("Loaded Content-Based model")

        if cf_model and content_model:
            hybrid_path = os.path.join(model_dir, 'hybrid_model.joblib')
            if os.path.exists(hybrid_path):
                hybrid_model = HybridRecommender.load(hybrid_path, cf_model, content_model)
                logging.info("Loaded Hybrid model")

    except Exception as e:
        logging.error(f"Error loading models: {e}")


# Load models on cold start
load_models()


@app.route(route="recommend/{user_id}")
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger function to get article recommendations.

    URL: /api/recommend/{user_id}
    Query params:
        n: Number of recommendations (default: 5)
        method: cf, content, or hybrid (default: hybrid)
    """
    logging.info('Recommendation request received')

    try:
        # Get user_id from route
        user_id = req.route_params.get('user_id')
        if not user_id:
            return func.HttpResponse(
                json.dumps({'error': 'user_id is required'}),
                status_code=400,
                mimetype='application/json'
            )

        user_id = int(user_id)

        # Get query parameters
        n = int(req.params.get('n', 5))
        method = req.params.get('method', 'hybrid').lower()

        # Select model
        if method == 'cf' and cf_model:
            model = cf_model
        elif method == 'content' and content_model:
            model = content_model
        elif method == 'hybrid' and hybrid_model:
            model = hybrid_model
        elif cf_model:
            model = cf_model
            method = 'cf'
        elif content_model:
            model = content_model
            method = 'content'
        else:
            return func.HttpResponse(
                json.dumps({'error': 'No model available'}),
                status_code=500,
                mimetype='application/json'
            )

        # Get recommendations
        recommendations = model.recommend(user_id, n_recommendations=n)

        result = {
            'user_id': user_id,
            'method': method,
            'recommendations': [
                {'article_id': int(aid), 'score': float(score)}
                for aid, score in recommendations
            ],
            'count': len(recommendations)
        }

        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype='application/json'
        )

    except ValueError as e:
        return func.HttpResponse(
            json.dumps({'error': f'Invalid user_id: {str(e)}'}),
            status_code=400,
            mimetype='application/json'
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype='application/json'
        )


@app.route(route="users")
def get_users(req: func.HttpRequest) -> func.HttpResponse:
    """Get list of available user IDs."""
    logging.info('Users list request received')

    try:
        if cf_model is None:
            return func.HttpResponse(
                json.dumps({'error': 'Model not loaded'}),
                status_code=500,
                mimetype='application/json'
            )

        limit = int(req.params.get('limit', 100))
        users = [int(uid) for uid in cf_model.user_id_map.keys()][:limit]

        return func.HttpResponse(
            json.dumps({
                'users': users,
                'total': len(cf_model.user_id_map),
                'returned': len(users)
            }),
            status_code=200,
            mimetype='application/json'
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype='application/json'
        )


@app.route(route="health")
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint."""
    return func.HttpResponse(
        json.dumps({
            'status': 'healthy',
            'models_loaded': {
                'cf': cf_model is not None,
                'content': content_model is not None,
                'hybrid': hybrid_model is not None
            }
        }),
        status_code=200,
        mimetype='application/json'
    )

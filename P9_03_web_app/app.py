"""
Flask Web Application for Article Recommendation System
Simple interface to get article recommendations for users

Run with: python app.py
"""

import os
import requests as http_requests
from flask import Flask, render_template, jsonify, request

# Configuration
API_URL = os.environ.get('API_URL', 'http://localhost:5001')

app = Flask(__name__)


def api_request(endpoint, params=None, timeout=10):
    """Make a request to the recommendation API."""
    try:
        response = http_requests.get(
            f"{API_URL}{endpoint}",
            params=params,
            timeout=timeout
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, response.json().get('error', 'Unknown error')
    except http_requests.exceptions.ConnectionError:
        return None, 'API not available. Make sure the API is running.'
    except Exception as e:
        return None, str(e)


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', api_url=API_URL)


@app.route('/api/health')
def health_check():
    """Check API health status."""
    data, error = api_request('/health')
    if data:
        return jsonify({'status': 'connected', 'api_data': data})
    return jsonify({'status': 'disconnected', 'error': error}), 503


@app.route('/api/users')
def get_users():
    """Proxy to get users from recommendation API."""
    limit = request.args.get('limit', 200, type=int)
    data, error = api_request('/users', params={'limit': limit})
    if data:
        return jsonify(data)
    return jsonify({'error': error}), 500


@app.route('/api/recommend/<int:user_id>')
def get_recommendations(user_id):
    """Proxy to get recommendations from recommendation API."""
    method = request.args.get('method', 'hybrid')
    n = request.args.get('n', 5, type=int)
    data, error = api_request(
        f'/recommend/{user_id}',
        params={'method': method, 'n': n},
        timeout=30
    )
    if data:
        return jsonify(data)
    return jsonify({'error': error}), 500


@app.route('/api/similar/<int:article_id>')
def get_similar(article_id):
    """Proxy to get similar articles from recommendation API."""
    n = request.args.get('n', 5, type=int)
    data, error = api_request(
        f'/similar/{article_id}',
        params={'n': n}
    )
    if data:
        return jsonify(data)
    return jsonify({'error': error}), 500


if __name__ == '__main__':
    print(f"Starting web app on http://localhost:5002")
    print(f"Recommendation API URL: {API_URL}")
    app.run(host='0.0.0.0', port=5002, debug=True)

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.recommendation_model import MovieRecommendationModel
import os

app = Flask(__name__)
CORS(app)

# Load data
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def load_sample_data():
    """Load sample movie data"""
    movies_data = {
        'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [
            'The Shawshank Redemption',
            'The Godfather',
            'The Dark Knight',
            'Pulp Fiction',
            'Forrest Gump',
            'Inception',
            'Interstellar',
            'The Matrix',
            'Parasite',
            'The Avengers'
        ],
        'genres': [
            'Drama, Crime',
            'Crime, Drama',
            'Action, Crime, Drama',
            'Crime, Drama',
            'Drama, Romance',
            'Action, Sci-Fi, Thriller',
            'Adventure, Drama, Sci-Fi',
            'Action, Sci-Fi',
            'Drama, Thriller',
            'Action, Adventure, Sci-Fi'
        ],
        'description': [
            'Two imprisoned men bond over a number of years',
            'The aging patriarch of an organized crime dynasty',
            'When the menace known as the Joker wreaks havoc',
            'The lives of two mob hitmen, a boxer and a gangster',
            'The presidencies of Kennedy and Johnson unfold',
            'A thief who steals corporate secrets through dream-sharing',
            'A team of explorers travel through a wormhole in space',
            'A computer hacker learns about the true nature of reality',
            'Greed and class conflict unfold within a South Korean family',
            'Earth\'s mightiest heroes must come together and learn to fight'
        ],
        'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.0]
    }
    return pd.DataFrame(movies_data)


# Initialize model
movies_df = load_sample_data()
model = MovieRecommendationModel(movies_df)
model.train()

# Routes

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'AI Movie Recommendation System',
        'version': '1.0.0',
        'endpoints': [
            'GET /movies - Daftar semua film',
            'POST /recommend/content - Rekomendasi berbasis konten',
            'POST /recommend/hybrid - Rekomendasi hybrid',
            'GET /movies/<movie_id> - Detail film'
        ]
    })


@app.route('/movies', methods=['GET'])
def get_all_movies():
    """Dapatkan daftar semua film"""
    movies_list = []
    for movie_id, row in movies_df.iterrows():
        movies_list.append({
            'id': int(movie_id),
            'title': row['title'],
            'genres': row['genres'],
            'rating': float(row['rating'])
        })
    return jsonify(movies_list)


@app.route('/movies/<int:movie_id>', methods=['GET'])
def get_movie_detail(movie_id):
    """Dapatkan detail film"""
    if movie_id not in movies_df.index:
        return jsonify({'error': 'Film tidak ditemukan'}), 404
    
    movie = movies_df.loc[movie_id]
    return jsonify({
        'id': movie_id,
        'title': movie['title'],
        'genres': movie['genres'],
        'description': movie['description'],
        'rating': float(movie['rating'])
    })


@app.route('/recommend/content', methods=['POST'])
def recommend_content_based():
    """
    Rekomendasi berbasis konten (Content-Based Filtering)
    
    Request JSON:
    {
        "movie_id": 1,
        "num_recommendations": 5
    }
    """
    data = request.get_json()
    
    if not data or 'movie_id' not in data:
        return jsonify({'error': 'movie_id diperlukan'}), 400
    
    movie_id = data['movie_id']
    n_recommendations = data.get('num_recommendations', 5)
    
    try:
        recommendations = model.get_content_based_recommendations(
            movie_id, n_recommendations
        )
        
        result = {
            'reference_movie': movies_df.loc[movie_id, 'title'],
            'recommendations': [
                {
                    'title': title,
                    'similarity_score': round(score, 2),
                    'movie_id': int(mid)
                }
                for title, score, mid in recommendations
            ]
        }
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/recommend/hybrid', methods=['POST'])
def recommend_hybrid():
    """
    Rekomendasi hybrid (Content-Based + Collaborative)
    
    Request JSON:
    {
        "movie_id": 1,  // optional
        "user_id": 1,   // optional
        "num_recommendations": 5
    }
    """
    data = request.get_json()
    
    if not data or ('movie_id' not in data and 'user_id' not in data):
        return jsonify({
            'error': 'Minimal harus ada movie_id atau user_id'
        }), 400
    
    movie_id = data.get('movie_id')
    user_id = data.get('user_id')
    n_recommendations = data.get('num_recommendations', 5)
    
    try:
        recommendations = model.get_hybrid_recommendations(
            user_id=user_id,
            movie_id=movie_id,
            n_recommendations=n_recommendations
        )
        
        result = {
            'recommendations': [
                {
                    'title': title,
                    'compatibility_score': round(score, 2),
                    'movie_id': int(mid),
                    'breakdown': {
                        'content_based': round(scores.get('content', 0), 2),
                        'collaborative': round(scores.get('collaborative', 0), 2)
                    }
                }
                for title, score, mid, scores in recommendations
            ]
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK',
        'model_trained': model.content_similarity is not None
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)

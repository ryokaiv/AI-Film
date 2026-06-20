import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class SimilarityCalculator:
    """Kalkulator untuk menghitung similarity score antar film"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.description_matrix = None
        self.genre_matrix = None
        
    def calculate_content_similarity(self, movies_df):
        """
        Hitung similarity berdasarkan konten film (deskripsi, genre, dll)
        
        Args:
            movies_df: DataFrame dengan kolom 'description' dan 'genres'
            
        Returns:
            similarity_matrix: Matrix similarity antar film
        """
        # TF-IDF untuk deskripsi
        descriptions = movies_df['description'].fillna('')
        self.description_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
        
        # Hitung similarity dari deskripsi
        desc_similarity = cosine_similarity(self.description_matrix)
        
        # One-hot encoding untuk genre
        genre_dummies = pd.get_dummies(
            movies_df['genres'].str.split(', ').explode()
        ).groupby(level=0).sum()
        
        # Normalize genre matrix
        genre_similarity = cosine_similarity(genre_dummies)
        
        # Gabungkan kedua similarity (weighted average)
        combined_similarity = (0.6 * desc_similarity + 0.4 * genre_similarity)
        
        return combined_similarity
    
    def calculate_user_similarity(self, ratings_df):
        """
        Hitung similarity antar pengguna berdasarkan rating mereka
        
        Args:
            ratings_df: DataFrame dengan kolom 'user_id', 'movie_id', 'rating'
            
        Returns:
            user_similarity_matrix: Matrix similarity antar pengguna
        """
        # Pivot untuk membuat user-item matrix
        user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        # Hitung cosine similarity
        user_similarity = cosine_similarity(user_item_matrix)
        
        return user_similarity, user_item_matrix
    
    def get_similarity_score(self, similarity_matrix, item_idx, reference_idx):
        """
        Dapatkan similarity score dalam persen (0-100%)
        
        Args:
            similarity_matrix: Matrix similarity
            item_idx: Index item yang dicek
            reference_idx: Index referensi
            
        Returns:
            similarity_score: Score dalam persen (0-100)
        """
        score = similarity_matrix[item_idx, reference_idx]
        # Normalize dari [-1, 1] ke [0, 100] jika diperlukan
        return max(0, min(100, (score + 1) / 2 * 100))
    
    def rank_by_similarity(self, similarity_scores, top_n=5):
        """
        Ranking items berdasarkan similarity score
        
        Args:
            similarity_scores: Array similarity scores
            top_n: Jumlah top items yang dikembalikan
            
        Returns:
            ranked_indices: Indeks items yang sudah diurutkan
            ranked_scores: Scores yang sudah diurutkan
        """
        sorted_indices = np.argsort(similarity_scores)[::-1][:top_n]
        sorted_scores = similarity_scores[sorted_indices]
        
        return sorted_indices, sorted_scores

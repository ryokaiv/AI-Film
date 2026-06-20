import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from .similarity import SimilarityCalculator
from .user_profile import UserProfile, UserProfileManager


class MovieRecommendationModel:
    """Model rekomendasi film dengan content-based dan collaborative filtering"""
    
    def __init__(self, movies_df):
        """
        Inisialisasi model
        
        Args:
            movies_df: DataFrame dengan kolom [movie_id, title, genres, description, rating]
        """
        self.movies_df = movies_df.set_index('movie_id')
        self.similarity_calc = SimilarityCalculator()
        self.user_manager = UserProfileManager()
        self.content_similarity = None
        self.user_similarity = None
        self.user_item_matrix = None
        
    def train(self, ratings_df=None):
        """
        Train model dengan menghitung similarity matrices
        
        Args:
            ratings_df: Optional DataFrame dengan [user_id, movie_id, rating]
        """
        # Hitung content-based similarity
        self.content_similarity = self.similarity_calc.calculate_content_similarity(
            self.movies_df
        )
        
        # Hitung collaborative filtering jika ada rating data
        if ratings_df is not None and len(ratings_df) > 0:
            (self.user_similarity, 
             self.user_item_matrix) = self.similarity_calc.calculate_user_similarity(ratings_df)
            
            # Load user profiles
            self.user_manager.bulk_load_from_csv_df(ratings_df)
    
    def get_content_based_recommendations(self, movie_id, n_recommendations=5):
        """
        Dapatkan rekomendasi berdasarkan film yang mirip (content-based)
        
        Args:
            movie_id: ID film referensi
            n_recommendations: Jumlah rekomendasi
            
        Returns:
            List of (movie_title, similarity_score, movie_id)
        """
        if self.content_similarity is None:
            raise ValueError("Model belum di-train. Jalankan train() terlebih dahulu.")
        
        # Cari index film
        if movie_id not in self.movies_df.index:
            raise ValueError(f"Film dengan ID {movie_id} tidak ditemukan")
        
        movie_idx = list(self.movies_df.index).index(movie_id)
        similarities = self.content_similarity[movie_idx]
        
        # Exclude film referensi sendiri
        similarities[movie_idx] = -1
        
        # Ambil top N
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            actual_movie_id = self.movies_df.index[idx]
            title = self.movies_df.loc[actual_movie_id, 'title']
            score = self.similarity_calc.get_similarity_score(
                self.content_similarity, movie_idx, idx
            )
            recommendations.append((title, score, actual_movie_id))
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """
        Dapatkan rekomendasi berdasarkan pengguna serupa (collaborative filtering)
        
        Args:
            user_id: ID pengguna
            n_recommendations: Jumlah rekomendasi
            
        Returns:
            List of (movie_title, predicted_rating, similarity_score)
        """
        if self.user_item_matrix is None:
            raise ValueError("Tidak ada data collaborative filtering. Load ratings terlebih dahulu.")
        
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"Pengguna dengan ID {user_id} tidak ditemukan")
        
        user_idx = list(self.user_item_matrix.index).index(user_id)
        user_similarities = self.user_similarity[user_idx]
        
        # Film yang sudah ditonton user
        watched = set(self.user_item_matrix.columns[
            self.user_item_matrix.iloc[user_idx] > 0
        ])
        
        # Prediksi rating untuk film yang belum ditonton
        predictions = {}
        for movie_id in self.user_item_matrix.columns:
            if movie_id not in watched:
                # Weighted average rating dari pengguna serupa
                similar_ratings = self.user_item_matrix[movie_id] * user_similarities
                similar_count = np.sum(self.user_item_matrix[movie_id] > 0)
                
                if similar_count > 0:
                    predicted_rating = np.sum(similar_ratings) / np.sum(user_similarities)
                    predictions[movie_id] = predicted_rating
        
        # Sort by predicted rating
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        recommendations = []
        for movie_id, pred_rating in sorted_predictions:
            if movie_id in self.movies_df.index:
                title = self.movies_df.loc[movie_id, 'title']
                recommendations.append((title, pred_rating, movie_id))
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id=None, movie_id=None, n_recommendations=5):
        """
        Dapatkan rekomendasi hybrid (gabung content-based + collaborative)
        
        Args:
            user_id: ID pengguna (opsional)
            movie_id: ID film (opsional)
            n_recommendations: Jumlah rekomendasi
            
        Returns:
            List of (movie_title, combined_score, details_dict)
        """
        recommendations = {}
        
        # Content-based recommendations
        if movie_id:
            try:
                content_recs = self.get_content_based_recommendations(
                    movie_id, n_recommendations * 2
                )
                for title, score, mid in content_recs:
                    if mid not in recommendations:
                        recommendations[mid] = {'title': title, 'scores': {}}
                    recommendations[mid]['scores']['content'] = score
            except:
                pass
        
        # Collaborative recommendations
        if user_id and self.user_item_matrix is not None:
            try:
                collab_recs = self.get_collaborative_recommendations(
                    user_id, n_recommendations * 2
                )
                for title, pred_rating, mid in collab_recs:
                    if mid not in recommendations:
                        recommendations[mid] = {'title': title, 'scores': {}}
                    # Normalize predicted rating ke 0-100
                    recommendations[mid]['scores']['collaborative'] = pred_rating * 20
            except:
                pass
        
        # Calculate combined score
        final_recs = []
        for movie_id, data in recommendations.items():
            scores = data['scores']
            
            # Weight: 60% content-based, 40% collaborative
            if 'content' in scores and 'collaborative' in scores:
                combined = 0.6 * scores['content'] + 0.4 * scores['collaborative']
            elif 'content' in scores:
                combined = scores['content']
            else:
                combined = scores.get('collaborative', 0)
            
            final_recs.append((data['title'], combined, movie_id, scores))
        
        # Sort dan ambil top N
        final_recs.sort(key=lambda x: x[1], reverse=True)
        return final_recs[:n_recommendations]
    
    def explain_recommendation(self, recommendation_tuple):
        """
        Jelaskan mengapa film direkomendasikan
        
        Args:
            recommendation_tuple: Tuple dari recommendation result
            
        Returns:
            String explanation
        """
        title, score, movie_id, scores = recommendation_tuple
        
        explanation = f"\n📽️ {title}\n"
        explanation += f"🎯 Tingkat Kecocokan: {score:.1f}%\n"
        
        if 'content' in scores:
            explanation += f"  • Kesamaan konten: {scores['content']:.1f}%\n"
        
        if 'collaborative' in scores:
            explanation += f"  • Rating prediksi: {scores['collaborative']:.1f}/100\n"
        
        return explanation

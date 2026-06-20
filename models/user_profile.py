import numpy as np
import pandas as pd


class UserProfile:
    """Mengelola profil dan preferensi pengguna"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.ratings = {}  # {movie_id: rating}
        self.watched_movies = set()
        self.genre_preferences = {}  # {genre: average_rating}
        self.average_rating = 0
        
    def add_rating(self, movie_id, rating):
        """Tambah atau update rating untuk film"""
        self.ratings[movie_id] = rating
        self.watched_movies.add(movie_id)
        
    def update_genre_preferences(self, movies_df):
        """
        Update preferensi genre berdasarkan rating
        
        Args:
            movies_df: DataFrame dengan kolom 'movie_id' dan 'genres'
        """
        genre_ratings = {}
        genre_counts = {}
        
        for movie_id, rating in self.ratings.items():
            if movie_id in movies_df.index:
                genres = movies_df.loc[movie_id, 'genres'].split(', ')
                for genre in genres:
                    if genre not in genre_ratings:
                        genre_ratings[genre] = 0
                        genre_counts[genre] = 0
                    genre_ratings[genre] += rating
                    genre_counts[genre] += 1
        
        # Hitung rata-rata rating per genre
        self.genre_preferences = {
            genre: genre_ratings[genre] / genre_counts[genre]
            for genre in genre_ratings
        }
        
        # Hitung average rating overall
        if self.ratings:
            self.average_rating = np.mean(list(self.ratings.values()))
    
    def get_preferred_genres(self, top_n=5):
        """
        Dapatkan top genre yang disukai pengguna
        
        Args:
            top_n: Jumlah top genre
            
        Returns:
            List of (genre, average_rating) sorted by rating
        """
        sorted_genres = sorted(
            self.genre_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_genres[:top_n]
    
    def get_rating_distribution(self):
        """
        Dapatkan distribusi rating pengguna
        
        Returns:
            Dictionary dengan {rating: count}
        """
        distribution = {}
        for rating in self.ratings.values():
            distribution[rating] = distribution.get(rating, 0) + 1
        return distribution
    
    def to_dict(self):
        """Konversi profil ke dictionary"""
        return {
            'user_id': self.user_id,
            'total_ratings': len(self.ratings),
            'average_rating': self.average_rating,
            'genre_preferences': self.genre_preferences,
            'watched_movies': list(self.watched_movies)
        }


class UserProfileManager:
    """Mengelola multiple user profiles"""
    
    def __init__(self):
        self.profiles = {}  # {user_id: UserProfile}
        
    def create_profile(self, user_id):
        """Buat profil baru untuk pengguna"""
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id)
        return self.profiles[user_id]
    
    def get_profile(self, user_id):
        """Dapatkan profil pengguna"""
        return self.profiles.get(user_id)
    
    def add_user_ratings(self, user_id, ratings_dict):
        """
        Tambah multiple ratings sekaligus
        
        Args:
            user_id: ID pengguna
            ratings_dict: {movie_id: rating}
        """
        profile = self.create_profile(user_id)
        for movie_id, rating in ratings_dict.items():
            profile.add_rating(movie_id, rating)
    
    def bulk_load_from_csv(self, csv_path):
        """
        Load ratings dari CSV file
        
        Args:
            csv_path: Path ke file CSV dengan kolom [user_id, movie_id, rating]
        """
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            rating = row['rating']
            
            profile = self.create_profile(user_id)
            profile.add_rating(movie_id, rating)

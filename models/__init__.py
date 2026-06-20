"""
Movie Recommendation System Models Package
"""

from .similarity import SimilarityCalculator
from .user_profile import UserProfile, UserProfileManager
from .recommendation_model import MovieRecommendationModel

__all__ = [
    'SimilarityCalculator',
    'UserProfile',
    'UserProfileManager',
    'MovieRecommendationModel'
]

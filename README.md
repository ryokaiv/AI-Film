# AI Movie Recommendation System

Sistem AI yang menggunakan machine learning untuk memberikan rekomendasi film berdasarkan preferensi pengguna dengan tingkat kecocokan (similarity score).

## Fitur

- **Content-Based Filtering**: Merekomendasikan film berdasarkan genre, pemain, sutradara, dan deskripsi
- **Collaborative Filtering**: Memprediksi rating berdasarkan pengguna dengan preferensi serupa
- **Similarity Score**: Menampilkan tingkat kecocokan (0-100%) untuk setiap rekomendasi
- **User Preference Learning**: Belajar dari rating dan riwayat penonton pengguna
- **Multi-language Support**: Mendukung input dalam bahasa Indonesia dan Inggris

## Struktur Project

```
AI-model/
├── data/
│   ├── movies.csv              # Database film
│   ├── user_ratings.csv        # Rating pengguna
│   └── user_preferences.csv    # Preferensi pengguna
├── models/
│   ├── recommendation_model.py # Model rekomendasi utama
│   ├── similarity.py           # Kalkulasi similarity score
│   └── user_profile.py         # Profil pengguna
├── app.py                      # Main application
├── requirements.txt            # Dependencies
└── README.md                   # Dokumentasi
```

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- flask (untuk API)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from app import MovieRecommender

recommender = MovieRecommender()
recommendations = recommender.get_recommendations(user_id=1, num_recommendations=5)

# Output: List of (movie_title, similarity_score)
```

## Model Architecture

### Content-Based Filtering
- TF-IDF untuk deskripsi film
- One-Hot Encoding untuk genre
- Cosine Similarity untuk menghitung kesamaan

### Collaborative Filtering
- User-Item Matrix
- Matrix Factorization (SVD)
- Prediksi rating berbasis item-item serupa

## Author

ryokaiv

## License

MIT

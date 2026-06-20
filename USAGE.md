# Usage Guide & Examples

Panduan lengkap cara menggunakan AI Movie Recommendation System.

## 1. Installation

```bash
# Clone repository
git clone https://github.com/ryokaiv/AI-model.git
cd AI-model

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Menjalankan Aplikasi

```bash
python app.py
```

Server akan berjalan di `http://localhost:5000`

## 3. API Endpoints

### 3.1 Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "OK",
  "model_trained": true
}
```

### 3.2 Daftar Semua Film
```bash
curl http://localhost:5000/movies
```

Response:
```json
[
  {
    "id": 1,
    "title": "The Shawshank Redemption",
    "genres": "Drama, Crime",
    "rating": 9.3
  },
  ...
]
```

### 3.3 Detail Film Spesifik
```bash
curl http://localhost:5000/movies/1
```

Response:
```json
{
  "id": 1,
  "title": "The Shawshank Redemption",
  "genres": "Drama, Crime",
  "description": "Two imprisoned men bond over a number of years",
  "rating": 9.3
}
```

### 3.4 Content-Based Recommendation

Dapatkan rekomendasi berdasarkan film yang mirip:

```bash
curl -X POST http://localhost:5000/recommend/content \
  -H "Content-Type: application/json" \
  -d '{
    "movie_id": 1,
    "num_recommendations": 5
  }'
```

Response:
```json
{
  "reference_movie": "The Shawshank Redemption",
  "recommendations": [
    {
      "title": "The Godfather",
      "similarity_score": 85.5,
      "movie_id": 2
    },
    {
      "title": "The Dark Knight",
      "similarity_score": 78.2,
      "movie_id": 3
    }
  ]
}
```

**Penjelasan:**
- `similarity_score`: Tingkat kesamaan konten film (0-100%)
- Semakin tinggi score, semakin mirip dengan film referensi

### 3.5 Hybrid Recommendation

Dapatkan rekomendasi gabungan (content-based + collaborative):

```bash
curl -X POST http://localhost:5000/recommend/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "movie_id": 1,
    "user_id": 1,
    "num_recommendations": 5
  }'
```

Response:
```json
{
  "recommendations": [
    {
      "title": "Inception",
      "compatibility_score": 87.3,
      "movie_id": 6,
      "breakdown": {
        "content_based": 82.1,
        "collaborative": 92.5
      }
    }
  ]
}
```

**Penjelasan:**
- `compatibility_score`: Tingkat kecocokan keseluruhan dengan selera user (0-100%)
- `content_based`: Skor berdasarkan kesamaan konten film
- `collaborative`: Skor berdasarkan preferensi pengguna serupa

## 4. Python Usage (Tanpa API)

### 4.1 Menggunakan Model Langsung

```python
import pandas as pd
from models.recommendation_model import MovieRecommendationModel

# Load data
movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/user_ratings.csv')

# Initialize model
model = MovieRecommendationModel(movies_df)
model.train(ratings_df)

# Get content-based recommendations
recommendations = model.get_content_based_recommendations(
    movie_id=1, 
    n_recommendations=5
)

for title, score, movie_id in recommendations:
    print(f"{title}: {score:.1f}% kesamaan")
```

### 4.2 Hybrid Recommendations

```python
# Get hybrid recommendations
recommendations = model.get_hybrid_recommendations(
    user_id=1,
    movie_id=1,
    n_recommendations=5
)

for title, score, movie_id, scores in recommendations:
    print(f"{title}: {score:.1f}% kecocokan")
    print(f"  - Content-based: {scores['content']:.1f}%")
    print(f"  - Collaborative: {scores['collaborative']:.1f}%")
```

### 4.3 User Profile Management

```python
from models.user_profile import UserProfileManager

manager = UserProfileManager()

# Create user profile
profile = manager.create_profile(user_id=1)

# Add ratings
profile.add_rating(movie_id=1, rating=5)
profile.add_rating(movie_id=2, rating=4)
profile.add_rating(movie_id=3, rating=3)

# Get profile info
print(profile.to_dict())

# Get preferred genres
preferred_genres = profile.get_preferred_genres(top_n=3)
for genre, avg_rating in preferred_genres:
    print(f"{genre}: {avg_rating:.1f}/5")
```

## 5. Algoritma yang Digunakan

### Content-Based Filtering
- **TF-IDF**: Untuk analisis deskripsi film
- **One-Hot Encoding**: Untuk kategori genre
- **Cosine Similarity**: Untuk menghitung kesamaan
- **Weighted Average**: 60% deskripsi + 40% genre

### Collaborative Filtering
- **User-Item Matrix**: Pivot table rating pengguna
- **Cosine Similarity**: Untuk menemukan pengguna serupa
- **Weighted Prediction**: Rating prediksi dari pengguna serupa

### Hybrid Approach
- **Combined Score**: 60% content-based + 40% collaborative

## 6. Customization

### Mengubah Bobot Hybrid

Edit di `models/recommendation_model.py` baris ~115:
```python
combined = 0.6 * scores['content'] + 0.4 * scores['collaborative']
```

### Menambah Data Film

Edit di `app.py` fungsi `load_sample_data()`:
```python
'title': ['Film 1', 'Film 2', ...],
'genres': ['Genre1, Genre2', ...],
...
```

### Menggunakan Data CSV

```python
movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/user_ratings.csv')

model = MovieRecommendationModel(movies_df)
model.train(ratings_df)
```

## 7. Troubleshooting

### Error: "Model belum di-train"
```python
model.train(ratings_df)  # Call train() sebelum get_recommendations()
```

### Error: "Film tidak ditemukan"
- Pastikan movie_id valid
- Check dengan GET `/movies` endpoint terlebih dahulu

### Error: "Pengguna tidak ditemukan"
- Pastikan user_id ada di ratings data
- atau gunakan hanya movie_id untuk content-based recommendation

## 8. Next Steps

- [ ] Tambahkan lebih banyak film ke database
- [ ] Load user ratings dari database
- [ ] Implementasikan feature saving/loading model
- [ ] Buat web interface (frontend)
- [ ] Deploy ke production (Heroku, AWS, dll)
- [ ] Tambahkan authentication & user management
- [ ] Implementasikan caching untuk performance

## 9. Performance Tips

1. **Cache similarity matrices** untuk film dataset besar
2. **Batch process** recommendations untuk banyak users
3. **Use async** untuk API calls
4. **Optimize** TF-IDF parameters

## 10. Support & Contact

Untuk pertanyaan atau bug report, buat issue di GitHub:
https://github.com/ryokaiv/AI-model/issues

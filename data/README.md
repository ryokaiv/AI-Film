# Data Directory

Direktori ini berisi data yang digunakan oleh sistem rekomendasi film.

## File-file

### movies.csv
Database film yang berisi:
- `movie_id`: ID unik film
- `title`: Judul film
- `genres`: Genre film (dipisahkan koma)
- `description`: Deskripsi sinopsis
- `rating`: Rating IMDB

### user_ratings.csv
Rating pengguna yang berisi:
- `user_id`: ID unik pengguna
- `movie_id`: ID film
- `rating`: Rating dari pengguna (1-5 atau 1-10)
- `timestamp`: Waktu rating (optional)

### user_preferences.csv
Preferensi pengguna yang berisi:
- `user_id`: ID unik pengguna
- `genre`: Genre favorit
- `preference_score`: Skor preferensi (0-1)

## Format CSV

Pastikan file CSV menggunakan encoding UTF-8 dan delimiter koma (,).

## Sample Data

Anda dapat menggunakan contoh data yang sudah disediakan di `app.py` dengan fungsi `load_sample_data()`.

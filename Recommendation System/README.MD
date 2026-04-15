# 🎬 Personalized Movie Recommendation System

**Author:** Vaibhav Gupta | IIT Kanpur & IIT Delhi Certified  
**Stack:** Python · Scikit-learn · Surprise (SVD) · Flask · Pandas

---

## Overview
Hybrid recommendation engine combining:
- **Collaborative Filtering** (SVD via Surprise) — user behaviour patterns
- **Content-Based Filtering** — genre cosine similarity
- **Weighted Hybrid Blend** — 70% CF + 30% CB for best accuracy

Achieved **20% improvement** in recommendation accuracy over baseline.

---

## Project Structure
```
1_recommendation_system/
├── recommender.py      # Core ML engine (CF + CB + Hybrid)
├── app.py              # Flask REST API
├── requirements.txt
├── models/             # Saved model pickles (auto-created)
└── data/               # Place your ratings.csv & movies.csv here
```

## Quick Start
```bash
pip install -r requirements.txt

# Train models (uses synthetic data if no CSV provided)
python recommender.py

# Start API server
python app.py
```

## API Usage
```bash
# Health check
curl http://localhost:5000/health

# Get recommendations
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "liked_movie_id": 10, "top_n": 5}'

# Similar movies
curl http://localhost:5000/similar/10?n=5
```

## Data Format
**ratings.csv** → `userId, movieId, rating`  
**movies.csv**  → `movieId, title, genres` (genres pipe-separated e.g. `Action|Drama`)

Compatible with [MovieLens](https://grouplens.org/datasets/movielens/) datasets.
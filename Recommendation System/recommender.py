"""
=============================================================
Personalized Movie Recommendation System
Author  : Vaibhav Gupta
Tech    : Python · Scikit-learn · Pandas · Flask · Surprise
=============================================================
Collaborative Filtering (SVD) + Content-Based hybrid engine
deployed as a REST API via Flask.
"""

import os
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════
def load_movielens(ratings_path: str, movies_path: str):
    """Load MovieLens-style ratings and movies CSVs."""
    ratings = pd.read_csv(ratings_path)
    movies  = pd.read_csv(movies_path)
    logger.info("Loaded %d ratings for %d movies", len(ratings), movies['movieId'].nunique())
    return ratings, movies


# ══════════════════════════════════════════════════════════
# 2. COLLABORATIVE FILTERING  (Surprise SVD)
# ══════════════════════════════════════════════════════════
class CollaborativeFilter:
    def __init__(self, n_factors: int = 100, n_epochs: int = 20, lr_all: float = 0.005, reg_all: float = 0.02):
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
        self.trained = False

    def train(self, ratings_df: pd.DataFrame) -> dict:
        reader  = Reader(rating_scale=(ratings_df["rating"].min(), ratings_df["rating"].max()))
        data    = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        self.model.fit(trainset)
        self.trained = True

        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae  = accuracy.mae(predictions,  verbose=False)
        logger.info("CF Model  RMSE=%.4f  MAE=%.4f", rmse, mae)
        return {"rmse": round(rmse, 4), "mae": round(mae, 4)}

    def predict(self, user_id: int, movie_id: int) -> float:
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(user_id, movie_id).est

    def top_n_for_user(self, user_id: int, all_movie_ids: list, n: int = 10) -> list:
        """Return top-N unseen movie predictions for a user."""
        preds = [(mid, self.predict(user_id, mid)) for mid in all_movie_ids]
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("CF model saved → %s", path)

    @staticmethod
    def load(path: str) -> "CollaborativeFilter":
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════
# 3. CONTENT-BASED FILTERING
# ══════════════════════════════════════════════════════════
class ContentBasedFilter:
    def __init__(self):
        self.mlb         = MultiLabelBinarizer()
        self.movie_matrix = None
        self.movies_df   = None

    def fit(self, movies_df: pd.DataFrame):
        """Build genre-based item feature matrix."""
        df = movies_df.copy()
        df["genres_list"] = df["genres"].apply(
            lambda g: g.split("|") if isinstance(g, str) else []
        )
        genre_matrix     = self.mlb.fit_transform(df["genres_list"])
        self.movie_matrix = pd.DataFrame(genre_matrix, index=df["movieId"])
        self.movies_df   = df.set_index("movieId")
        logger.info("CB Filter built for %d movies, %d genre features",
                    len(df), genre_matrix.shape[1])

    def similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        if movie_id not in self.movie_matrix.index:
            raise ValueError(f"movieId {movie_id} not found.")
        vec  = self.movie_matrix.loc[[movie_id]]
        sims = cosine_similarity(vec, self.movie_matrix)[0]
        sim_series = pd.Series(sims, index=self.movie_matrix.index).drop(movie_id)
        top_ids    = sim_series.nlargest(n).index
        result     = self.movies_df.loc[top_ids, ["title", "genres"]].copy()
        result["similarity"] = sims[self.movie_matrix.index.isin(top_ids)]
        return result.reset_index()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ContentBasedFilter":
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════
# 4. HYBRID ENGINE
# ══════════════════════════════════════════════════════════
class HybridRecommender:
    """
    Weighted blend of Collaborative and Content-Based scores.
    cf_weight + cb_weight should sum to 1.0
    """
    def __init__(self, cf: CollaborativeFilter, cb: ContentBasedFilter,
                 cf_weight: float = 0.7, cb_weight: float = 0.3):
        self.cf = cf
        self.cb = cb
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight

    def recommend(self, user_id: int, liked_movie_id: int,
                  all_movie_ids: list, n: int = 10) -> pd.DataFrame:
        # CF scores
        cf_preds = dict(self.cf.top_n_for_user(user_id, all_movie_ids, n=len(all_movie_ids)))

        # CB scores from liked movie
        cb_df    = self.cb.similar_movies(liked_movie_id, n=len(all_movie_ids))
        cb_scores = dict(zip(cb_df["movieId"], cb_df["similarity"]))

        # Blend
        blended = {}
        for mid in all_movie_ids:
            cf_s = cf_preds.get(mid, 0)
            cb_s = cb_scores.get(mid, 0)
            # normalise CF to [0,1]  (ratings are typically 0.5–5)
            cf_norm = cf_s / 5.0
            blended[mid] = self.cf_weight * cf_norm + self.cb_weight * cb_s

        top = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:n]
        return pd.DataFrame(top, columns=["movieId", "hybrid_score"])


# ══════════════════════════════════════════════════════════
# 5. TRAINING ENTRY-POINT
# ══════════════════════════════════════════════════════════
def train_and_save(ratings_path: str, movies_path: str):
    ratings, movies = load_movielens(ratings_path, movies_path)

    cf = CollaborativeFilter()
    metrics = cf.train(ratings)
    cf.save(str(MODEL_DIR / "cf_model.pkl"))

    cb = ContentBasedFilter()
    cb.fit(movies)
    cb.save(str(MODEL_DIR / "cb_model.pkl"))

    logger.info("Training complete. Metrics: %s", metrics)
    return cf, cb, metrics


if __name__ == "__main__":
    # ── Quick smoke-test with synthetic data ──────────────
    np.random.seed(42)
    n_users, n_movies = 200, 500
    ratings_sample = pd.DataFrame({
        "userId"  : np.random.randint(1, n_users + 1, 5000),
        "movieId" : np.random.randint(1, n_movies + 1, 5000),
        "rating"  : np.random.choice([1, 2, 3, 4, 5], 5000,
                                     p=[0.05, 0.10, 0.20, 0.35, 0.30]),
    }).drop_duplicates(["userId", "movieId"])

    genres_pool = ["Action", "Comedy", "Drama", "Thriller",
                   "Romance", "Sci-Fi", "Horror", "Animation"]
    movies_sample = pd.DataFrame({
        "movieId": range(1, n_movies + 1),
        "title"  : [f"Movie_{i}" for i in range(1, n_movies + 1)],
        "genres" : ["|".join(np.random.choice(genres_pool,
                    np.random.randint(1, 4), replace=False))
                    for _ in range(n_movies)],
    })

    cf = CollaborativeFilter(n_factors=50, n_epochs=10)
    metrics = cf.train(ratings_sample)
    print(f"\n✅ CF Metrics  →  RMSE: {metrics['rmse']}  MAE: {metrics['mae']}")

    cb = ContentBasedFilter()
    cb.fit(movies_sample)
    sim = cb.similar_movies(1, n=5)
    print(f"\n✅ Top-5 Content-similar to Movie_1:\n{sim[['movieId','title','similarity']].to_string(index=False)}")

    hybrid = HybridRecommender(cf, cb)
    recs = hybrid.recommend(user_id=1, liked_movie_id=1,
                            all_movie_ids=list(range(1, 101)), n=5)
    print(f"\n✅ Hybrid Recommendations for User 1:\n{recs.to_string(index=False)}")
    print("\n🎉  recommender.py — smoke-test PASSED")
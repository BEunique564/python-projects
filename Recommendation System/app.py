"""
Flask REST API — Personalized Movie Recommendation System
Author: Vaibhav Gupta
Endpoints:
  GET  /health              → service health check
  POST /recommend           → hybrid recommendations for a user
  GET  /similar/<movie_id>  → content-based similar movies
"""

import pickle
from pathlib import Path
from flask import Flask, request, jsonify
from recommender import CollaborativeFilter, ContentBasedFilter, HybridRecommender

app = Flask(__name__)
MODEL_DIR = Path(__file__).parent / "models"

# ── Load models at startup ─────────────────────────────────
def load_models():
    cf_path = MODEL_DIR / "cf_model.pkl"
    cb_path = MODEL_DIR / "cb_model.pkl"
    if not cf_path.exists() or not cb_path.exists():
        raise FileNotFoundError("Models not found. Run `python recommender.py` first.")
    with open(cf_path, "rb") as f:
        cf = pickle.load(f)
    with open(cb_path, "rb") as f:
        cb = pickle.load(f)
    return HybridRecommender(cf, cb)

try:
    hybrid = load_models()
    MODEL_READY = True
except FileNotFoundError as e:
    MODEL_READY = False
    print(f"[WARN] {e}")


# ── Routes ────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_READY}), 200


@app.route("/recommend", methods=["POST"])
def recommend():
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Train first."}), 503

    data = request.get_json(force=True)
    user_id       = data.get("user_id")
    liked_movie   = data.get("liked_movie_id")
    all_movies    = data.get("all_movie_ids", list(range(1, 501)))
    top_n         = int(data.get("top_n", 10))

    if user_id is None or liked_movie is None:
        return jsonify({"error": "user_id and liked_movie_id are required."}), 400

    try:
        recs = hybrid.recommend(int(user_id), int(liked_movie), all_movies, n=top_n)
        return jsonify({
            "user_id"         : user_id,
            "liked_movie_id"  : liked_movie,
            "recommendations" : recs.to_dict(orient="records"),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/similar/<int:movie_id>")
def similar(movie_id):
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded."}), 503
    n = int(request.args.get("n", 10))
    try:
        sim_df = hybrid.cb.similar_movies(movie_id, n=n)
        return jsonify({
            "movie_id"       : movie_id,
            "similar_movies" : sim_df.to_dict(orient="records"),
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
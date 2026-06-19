"""
AI Mood Sequencer - Flask backend
=================================
Serves the frontend and exposes a small JSON API backed by the trained
LSTM next-track predictor (models/lstm_next_track.keras).

The prediction pipeline mirrors the team's app.py:
  - predict the next mood vector from a sliding feature window
  - retrieve the nearest real track to that predicted vector
The window length is read directly from the model (input shape (None, W, 5)),
so it always matches whatever model is shipped in ../models.

Run:
    cd frontend
    pip install -r requirements.txt
    python server.py
Then open http://127.0.0.1:5000
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from scipy.spatial.distance import cdist
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------
# Paths (frontend/ lives inside the project root)
# ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, "models", "lstm_next_track.keras")
DATA_PATH = os.path.join(ROOT, "data", "spotify_tracks.csv")

# Feature order the model was trained on (same as app.py)
MOOD_FEATURES = ["energy", "valence", "danceability", "acousticness", "tempo"]
# Order used by the UI bars / labels: Energy, Valence, Tempo, Dance, Acoustic
DISPLAY_ORDER = ["energy", "valence", "tempo", "danceability", "acousticness"]
DISPLAY_LABELS = ["Energy", "Valence", "Tempo", "Dance", "Acoustic"]


def _load_model_compat(path):
    """Load the model, tolerating Keras version skew between teammates.

    Newer Keras writes a 'quantization_config' key into layer configs that
    older Keras can't parse. If a plain load fails for that reason, retry with
    a tiny shim that drops the unknown kwarg.
    """
    try:
        return load_model(path)
    except (TypeError, ValueError):
        import keras
        _orig = keras.layers.Layer.__init__

        def _patched(self, *a, **k):
            k.pop("quantization_config", None)
            return _orig(self, *a, **k)

        keras.layers.Layer.__init__ = _patched
        try:
            return load_model(path)
        finally:
            keras.layers.Layer.__init__ = _orig


# ------------------------------------------------------------------
# Load model + data once at startup
# ------------------------------------------------------------------
print("Loading model:", MODEL_PATH)
model = _load_model_compat(MODEL_PATH)
WINDOW = int(model.input_shape[1])          # e.g. 7
print("Model window size:", WINDOW)

print("Loading tracks:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["track_name", "artists"] + MOOD_FEATURES)
df["artist"] = df["artists"].astype(str).str.split(";").str[0]
# The Spotify dataset repeats the same song across genres. De-duplicate so a
# playlist never lists the same (track, artist) twice; keep the most popular copy.
df = (df.sort_values("popularity", ascending=False)
        .drop_duplicates(subset=["track_name", "artist"], keep="first")
        .reset_index(drop=True))
df["display_name"] = df["track_name"].astype(str) + " - " + df["artist"]

TEMPO_MAX = float(df["tempo"].max()) or 1.0

GENRES = sorted(df["track_genre"].dropna().unique().tolist())

app = Flask(__name__, static_folder=HERE, static_url_path="")
CORS(app)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def display_vector(row):
    """Return features in UI order, all scaled to 0-1 (tempo normalised)."""
    return [
        round(float(row["energy"]), 3),
        round(float(row["valence"]), 3),
        round(min(float(row["tempo"]) / TEMPO_MAX, 1.0), 3),
        round(float(row["danceability"]), 3),
        round(float(row["acousticness"]), 3),
    ]


def norm_matrix(rows_df):
    """Feature matrix in 0-1 space (tempo normalised) for retrieval / scoring."""
    m = rows_df[DISPLAY_ORDER].astype(float).copy()
    m["tempo"] = (m["tempo"] / TEMPO_MAX).clip(0, 1)
    return m.values


def track_payload(row):
    return {
        "n": str(row["track_name"]),
        "a": str(row["artist"]),
        "g": str(row["track_genre"]),
        "f": display_vector(row),
    }


def predict_next_features(window_matrix):
    X = np.array(window_matrix, dtype=np.float32)
    X = np.expand_dims(X, axis=0)
    return model.predict(X, verbose=0)[0]


def find_closest_idx(vector, pool_df):
    vecs = pool_df[MOOD_FEATURES].astype(float).values
    d = cdist([vector], vecs, metric="euclidean")[0]
    return pool_df.index[int(np.argmin(d))]


def smoothness_score(feature_matrix):
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    if len(feature_matrix) < 2:
        return 0.0
    diffs = np.linalg.norm(np.diff(feature_matrix, axis=0), axis=1)
    return float(np.mean(diffs))


def build_seed_window(genre_df, seed_idx, size):
    """The `size` nearest tracks to the seed, with distinct titles so the
    opening of every playlist isn't several covers of the same song."""
    seed_vec = genre_df.loc[seed_idx, MOOD_FEATURES].astype(float).values
    vecs = genre_df[MOOD_FEATURES].astype(float).values
    d = cdist([seed_vec], vecs, metric="euclidean")[0]
    order = genre_df.index[np.argsort(d)]
    picked, titles = [], set()
    for idx in order:
        title = genre_df.loc[idx, "track_name"]
        if title in titles:
            continue
        titles.add(title)
        picked.append(idx)
        if len(picked) == size:
            break
    return picked


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
@app.route("/api/meta")
def meta():
    return jsonify({
        "genres": GENRES,
        "labels": DISPLAY_LABELS,
        "window": WINDOW,
        "track_count": int(len(df)),
    })


@app.route("/api/search")
def search():
    q = (request.args.get("q") or "").strip().lower()
    if not q:
        return jsonify([])
    mask = df["track_name"].astype(str).str.lower().str.contains(q, regex=False)
    hits = df[mask].sort_values("popularity", ascending=False).head(8)
    return jsonify([track_payload(r) for _, r in hits.iterrows()])


@app.route("/api/similar", methods=["POST"])
def similar():
    body = request.get_json(force=True)
    name = str(body.get("name", ""))
    artist = str(body.get("artist", ""))
    match = df[(df["track_name"].astype(str) == name)]
    if artist:
        m2 = match[match["artist"].astype(str) == artist]
        if len(m2):
            match = m2
    if not len(match):
        return jsonify({"selected": None, "similar": []})
    seed = match.iloc[0]
    # similarity uses 0-1 normalised features so mood (not raw BPM) drives it
    vecs = norm_matrix(df)
    seed_vec = norm_matrix(df.loc[[seed.name]])[0]
    d = cdist([seed_vec], vecs, metric="euclidean")[0]
    order = np.argsort(d)
    out, seen_titles = [], set()
    for pos in order:
        row = df.iloc[pos]
        title = str(row["track_name"])
        if title == name or title in seen_titles:
            continue
        seen_titles.add(title)
        out.append(track_payload(row))
        if len(out) >= 12:
            break
    return jsonify({"selected": track_payload(seed), "similar": out})


@app.route("/api/songs")
def songs():
    genre = request.args.get("genre", "")
    gdf = df[df["track_genre"] == genre].sort_values("popularity", ascending=False)
    names = gdf["display_name"].head(300).tolist()
    return jsonify(names)


@app.route("/api/journey", methods=["POST"])
def journey():
    body = request.get_json(force=True)
    genre = str(body.get("genre", ""))
    length = int(body.get("length", 8))
    pool_size = int(body.get("pool", 300))
    start_song = body.get("song")  # optional display_name

    genre_df = df[df["track_genre"] == genre].copy()
    if len(genre_df) < WINDOW + 1:
        return jsonify({"error": "Not enough tracks in this genre."}), 400

    length = max(length, WINDOW + 1)

    # seed
    if start_song and (genre_df["display_name"] == start_song).any():
        seed_idx = genre_df[genre_df["display_name"] == start_song].index[0]
    else:
        seed_idx = genre_df.sample(1).index[0]

    seed_ids = build_seed_window(genre_df, seed_idx, WINDOW)
    playlist_ids = list(seed_ids)
    used_titles = set(genre_df.loc[seed_ids, "track_name"])
    current_window = genre_df.loc[seed_ids, MOOD_FEATURES].astype(float).values

    remaining = genre_df.drop(playlist_ids, errors="ignore")
    remaining = remaining[~remaining["track_name"].isin(used_titles)]
    candidate_pool = (remaining.sample(min(pool_size, len(remaining)), random_state=42).copy()
                      if len(remaining) else remaining)

    while len(playlist_ids) < length and len(candidate_pool):
        pred = predict_next_features(current_window)
        nxt_idx = find_closest_idx(pred, candidate_pool)
        title = candidate_pool.loc[nxt_idx, "track_name"]
        nxt_feat = candidate_pool.loc[nxt_idx, MOOD_FEATURES].astype(float).values
        playlist_ids.append(nxt_idx)
        # drop the chosen track and any same-title cover
        candidate_pool = candidate_pool[candidate_pool["track_name"] != title]
        current_window = np.vstack([current_window[1:], nxt_feat])

    playlist = df.loc[playlist_ids]

    gen_score = smoothness_score(norm_matrix(playlist))
    rnd = genre_df.sample(min(len(playlist), len(genre_df)), random_state=123)
    rnd_score = smoothness_score(norm_matrix(rnd))
    ratio = (rnd_score / gen_score) if gen_score else 0.0

    return jsonify({
        "genre": genre,
        "tracks": [track_payload(r) for _, r in playlist.iterrows()],
        "generated_score": round(gen_score, 3),
        "random_score": round(rnd_score, 3),
        "ratio": round(ratio, 2),
    })


# ------------------------------------------------------------------
# Static frontend
# ------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

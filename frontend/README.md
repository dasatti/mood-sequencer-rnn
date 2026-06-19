# AI Mood Sequencer — Frontend

A web UI for the Mood Sequencer LSTM, served by a small Flask backend that
runs the **trained model** (`../models/lstm_next_track.keras`) against the
track catalogue (`../data/spotify_tracks.csv`).

Two modes:

- **Discover** — search a track, see its mood features and the nearest tracks by mood.
- **Mood Journey** — pick a genre; the LSTM predicts the next mood vector step by step
  and retrieves the closest real track each time, producing a smooth playlist. A
  smoothness score is compared against a random playlist of the same length.

## Run locally

```bash
cd frontend
pip install -r requirements.txt
python server.py
```

Then open <http://127.0.0.1:5000>.

The server reads the model and CSV from the project root (`../models`, `../data`),
so run it from inside the `frontend/` folder of the repo.

## How it maps to the team pipeline

The API mirrors `app.py`'s logic (`build_seed_window` → `predict_next_features`
→ `find_closest_track`). The sliding-window length is read straight from the
model's input shape (`model.input_shape[1]`), so it always matches the shipped
model — the current model expects a window of **7** frames of **5** features
(`energy, valence, danceability, acousticness, tempo`).

## API

| Method | Path           | Body / Query              | Returns |
|--------|----------------|---------------------------|---------|
| GET    | `/api/meta`    | —                         | genres, window size, track count |
| GET    | `/api/search`  | `?q=`                     | up to 8 matching tracks |
| POST   | `/api/similar` | `{name, artist}`          | selected track + 12 nearest |
| GET    | `/api/songs`   | `?genre=`                 | song display names in genre |
| POST   | `/api/journey` | `{genre, length, song?}`  | ordered playlist + smoothness scores |

Track feature vectors returned to the UI are ordered `[Energy, Valence, Tempo,
Dance, Acoustic]` and scaled to 0–1 (tempo normalised) for display.

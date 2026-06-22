# AI Mood Sequencer

> An LSTM that sequences songs into one smooth, continuous **mood journey** — instead of a random shuffle.

Most playlists and shuffle algorithms ignore *flow*: a calm acoustic track can slam straight into high-energy EDM. **AI Mood Sequencer** treats a playlist as a sequence (like words in a sentence) and uses a trained LSTM to predict the *mood* of the next track, then retrieves the closest real song. Chaining these predictions produces a playlist whose energy and emotion drift smoothly from start to finish.

Built on a catalogue of **114,000 Spotify tracks** across **114 genres**.

---

## How it works

The pipeline predicts the next mood, retrieves the nearest real track, and repeats:

1. **Data** — every track is reduced to 5 mood features: `energy`, `valence`, `danceability`, `acousticness`, `tempo`.
2. **Sequences** — a sliding window over genre-grouped playlists builds training examples (the shipped model uses a window of **7** tracks).
3. **LSTM** — two stacked LSTM layers (64 → 32) followed by dense layers output the next track's predicted mood vector (5 values).
4. **Retrieve & repeat** — find the nearest real track to that vector (Euclidean distance), append it, slide the window forward, and continue until the playlist is full.
5. **Feedback loop** — each chosen track conditions the next prediction, keeping the whole journey coherent.

The result is evaluated with a **smoothness** score (mean step-to-step change in mood features — lower is smoother), compared against a random same-genre playlist of equal length.

---

## Repository structure

```
mood-sequencer-rnn/
├── app.py                       # Streamlit app (Mood Journey Playlist Generator)
├── playlist_generator.py        # entry point / glue (see utils/ for the logic)
├── utils/
│   └── playlist_generator.py    # core helpers: predict_next_features, find_closest_track,
│                                #   build_seed_window, generate_playlist, smoothness_score
├── notebooks/
│   ├── 00_preprocessing.ipynb       # clean the raw Spotify dataset
│   ├── 01_sequence_construction.ipynb  # build sliding-window training sequences
│   ├── 02_lstm_training.ipynb       # train the next-track LSTM
│   └── 03_playlist_generation.ipynb # generate & evaluate playlists
├── data/
│   ├── spotify_tracks.csv           # raw track catalogue (~114k tracks)
│   ├── sequences.json / .pkl        # constructed sequences
│   ├── training_sequences.npz       # windowed training data
│   ├── train_val_test_split.npz     # train / val / test split
│   ├── X.npy / y.npy                # model inputs and targets
├── models/
│   └── lstm_next_track.keras        # trained LSTM (input shape: 7 × 5)
├── frontend/                        # web app (Flask API + HTML/JS UI)
└── requirements.txt
```

---

## Dataset

The model is trained on a Spotify tracks dataset (~114,000 rows, 114 genres). Each track is represented by five audio/mood features used throughout the project:

| Feature        | Meaning                                              |
|----------------|------------------------------------------------------|
| `energy`       | Intensity and activity (0–1)                         |
| `valence`      | Musical positivity / happiness (0–1)                 |
| `danceability` | How suitable a track is for dancing (0–1)            |
| `acousticness` | Confidence the track is acoustic (0–1)               |
| `tempo`        | Estimated tempo in BPM                               |

---

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Streamlit app

The original interface — pick a genre and starting song, set playlist length, and generate a mood journey with smoothness comparison and feature charts.

```bash
streamlit run app.py
```

### 3. Run the web frontend (Flask + LSTM)

A polished web UI served by a Flask backend that runs the trained model on every request. It has two modes: **Discover** (search a track, see the closest songs by mood) and **Mood Journey** (pick a genre, the LSTM builds the playlist live).

```bash
cd frontend
pip install -r requirements.txt
python server.py                 # → http://127.0.0.1:5000
```

See [`frontend/README.md`](frontend/README.md) for the API reference and details.

---

## Reproducing the model

Run the notebooks in order:

1. `00_preprocessing.ipynb` — clean and prepare the raw dataset.
2. `01_sequence_construction.ipynb` — build sliding-window training sequences.
3. `02_lstm_training.ipynb` — train the LSTM and export `models/lstm_next_track.keras`.
4. `03_playlist_generation.ipynb` — generate playlists and evaluate smoothness.

---

## Tech stack

Python · TensorFlow / Keras · pandas · NumPy · scikit-learn · SciPy · Streamlit · Flask · matplotlib

---

## Team

Danish Satti · Abdul Saboor · Umer Zohaib

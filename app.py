
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
from scipy.spatial.distance import cdist


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Mood Journey Playlist Generator",
    page_icon="🎵",
    layout="wide"
)

MOOD_FEATURES = [
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "tempo"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_lstm_model():
    return load_model("models/lstm_next_track.keras")


@st.cache_data
def load_tracks():
    return pd.read_csv(
        "data/spotify_tracks.csv"
    )


model = load_lstm_model()

df = load_tracks()


# ============================================================
# PREPARE DISPLAY DATA
# ============================================================

df = df.reset_index(drop=True)

df["display_name"] = (
    df["track_name"].astype(str)
    + " - "
    + df["artists"].astype(str)
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_next_features(
    model,
    sequence_window
):

    X = np.array(
        sequence_window,
        dtype=np.float32
    )

    X = np.expand_dims(
        X,
        axis=0
    )

    prediction = model.predict(
        X,
        verbose=0
    )

    return prediction[0]


def find_closest_track(
    predicted_vector,
    candidate_pool
):

    candidate_vectors = (
        candidate_pool[
            MOOD_FEATURES
        ]
        .astype(float)
        .values
    )

    distances = cdist(
        [predicted_vector],
        candidate_vectors,
        metric="euclidean"
    )[0]

    best_idx = np.argmin(
        distances
    )

    return candidate_pool.iloc[
        best_idx
    ]


def smoothness_score(
    feature_matrix
):

    distances = []

    for i in range(
        len(feature_matrix) - 1
    ):

        distance = np.linalg.norm(
            feature_matrix[i + 1]
            -
            feature_matrix[i]
        )

        distances.append(distance)

    return np.mean(
        distances
    )


def build_seed_window(
    genre_df,
    selected_song,
    window_size=5
):
    """
    Build an initial sequence
    around the chosen song.
    """

    seed_row = genre_df[
        genre_df["display_name"]
        ==
        selected_song
    ].iloc[0]

    seed_vector = (
        seed_row[
            MOOD_FEATURES
        ]
        .astype(float)
        .values
    )

    candidate_vectors = (
        genre_df[
            MOOD_FEATURES
        ]
        .astype(float)
        .values
    )

    distances = cdist(
        [seed_vector],
        candidate_vectors,
        metric="euclidean"
    )[0]

    nearest_indices = np.argsort(
        distances
    )[:window_size]

    return genre_df.iloc[
        nearest_indices
    ].copy()


def generate_playlist(
    model,
    genre_df,
    selected_song,
    playlist_length,
    candidate_pool_size
):

    seed_window = build_seed_window(
        genre_df,
        selected_song,
        window_size=5
    )

    current_window = (
        seed_window[
            MOOD_FEATURES
        ]
        .astype(float)
        .values
    )

    playlist_track_ids = (
        seed_window.index.tolist()
    )

    candidate_pool = (
        genre_df.drop(
            playlist_track_ids,
            errors="ignore"
        )
        .sample(
            min(
                candidate_pool_size,
                len(genre_df)
                -
                len(playlist_track_ids)
            ),
            random_state=42
        )
        .copy()
    )

    while (
        len(playlist_track_ids)
        <
        playlist_length
    ):

        predicted_vector = (
            predict_next_features(
                model,
                current_window
            )
        )

        next_track = (
            find_closest_track(
                predicted_vector,
                candidate_pool
            )
        )

        next_track_id = (
            next_track.name
        )

        playlist_track_ids.append(
            next_track_id
        )

        candidate_pool = (
            candidate_pool.drop(
                next_track_id,
                errors="ignore"
            )
        )

        next_features = np.array(
            next_track[
                MOOD_FEATURES
            ],
            dtype=np.float32
        )

        current_window = np.vstack([
            current_window[1:],
            next_features
        ])

        if len(candidate_pool) == 0:
            break

    return df.loc[
        playlist_track_ids
    ]


# ============================================================
# UI
# ============================================================

st.title(
    "🎵 Mood Journey Playlist Generator"
)

st.markdown(
    """
    Generate smooth mood-transition playlists
    using a trained LSTM model.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Playlist Settings"
)

playlist_length = (
    st.sidebar.slider(
        "Playlist Length",
        min_value=5,
        max_value=15,
        value=8
    )
)

candidate_pool_size = (
    st.sidebar.slider(
        "Candidate Pool Size",
        min_value=100,
        max_value=1000,
        value=300,
        step=50
    )
)

selected_genre = (
    st.sidebar.selectbox(
        "Genre",
        sorted(
            df["track_genre"]
            .dropna()
            .unique()
        )
    )
)

genre_df = df[
    df["track_genre"]
    ==
    selected_genre
].copy()

selected_song = (
    st.sidebar.selectbox(
        "Starting Song",
        sorted(
            genre_df[
                "display_name"
            ].unique()
        )
    )
)


# ============================================================
# GENERATE
# ============================================================

if st.button(
    "Generate Playlist"
):

    with st.spinner(
        "Generating mood journey..."
    ):

        playlist_df = (
            generate_playlist(
                model=model,
                genre_df=genre_df,
                selected_song=selected_song,
                playlist_length=playlist_length,
                candidate_pool_size=candidate_pool_size
            )
        )

    # ========================================================
    # PLAYLIST
    # ========================================================

    st.subheader(
        "Generated Playlist"
    )

    display_cols = []

    for col in [
        "track_name",
        "artists",
        "track_genre",
        "popularity"
    ]:
        if col in playlist_df.columns:
            display_cols.append(col)

    st.dataframe(
        playlist_df[
            display_cols
        ],
        use_container_width=True
    )

    # ========================================================
    # SCORES
    # ========================================================

    generated_score = (
        smoothness_score(
            playlist_df[
                MOOD_FEATURES
            ].values
        )
    )

    random_playlist = (
        genre_df.sample(
            len(playlist_df),
            random_state=123
        )
    )

    random_score = (
        smoothness_score(
            random_playlist[
                MOOD_FEATURES
            ].values
        )
    )

    st.subheader(
        "Smoothness Comparison"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Generated Playlist",
            f"{generated_score:.4f}"
        )

    with col2:
        st.metric(
            "Random Playlist",
            f"{random_score:.4f}"
        )

    # ========================================================
    # MOOD JOURNEY
    # ========================================================

    st.subheader(
        "Mood Journey"
    )

    chart_df = pd.DataFrame({

        "Position":
            range(
                1,
                len(playlist_df) + 1
            ),

        "Energy":
            playlist_df[
                "energy"
            ].values,

        "Valence":
            playlist_df[
                "valence"
            ].values,

        "Danceability":
            playlist_df[
                "danceability"
            ].values

    })

    st.line_chart(
        chart_df.set_index(
            "Position"
        )
    )

    # ========================================================
    # FEATURE TABLE
    # ========================================================

    st.subheader(
        "Feature Progression"
    )

    st.dataframe(

        playlist_df[
            MOOD_FEATURES
        ].reset_index(
            drop=True
        ),

        use_container_width=True
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    csv_data = (
        playlist_df.to_csv(
            index=False
        )
    )

    st.download_button(
        label="Download Playlist CSV",
        data=csv_data,
        file_name="playlist.csv",
        mime="text/csv"
    )

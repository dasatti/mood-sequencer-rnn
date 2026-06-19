
import json
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
from scipy.spatial.distance import cdist

# =====================================================
# CONFIG
# =====================================================

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

# =====================================================
# HELPERS
# =====================================================

@st.cache_resource
def load_lstm_model():
    return load_model("models/lstm_next_track.keras")


@st.cache_data
def load_tracks():
    return pd.read_csv("data/spotify_tracks.csv")


@st.cache_data
def load_sequences():
    with open("data/sequences.json", "r") as f:
        return json.load(f)


def predict_next_features(model, sequence_window):

    X = np.array(
        sequence_window,
        dtype=np.float32
    )

    X = np.expand_dims(X, axis=0)

    prediction = model.predict(
        X,
        verbose=0
    )

    return prediction[0]


def find_closest_track(
    predicted_vector,
    candidate_pool
):

    candidate_vectors = candidate_pool[
        MOOD_FEATURES
    ].values

    distances = cdist(
        [predicted_vector],
        candidate_vectors,
        metric="euclidean"
    )[0]

    best_idx = np.argmin(distances)

    return candidate_pool.iloc[best_idx]


def smoothness_score(feature_matrix):

    distances = []

    for i in range(len(feature_matrix) - 1):

        distance = np.linalg.norm(
            feature_matrix[i + 1]
            -
            feature_matrix[i]
        )

        distances.append(distance)

    return np.mean(distances)


def build_candidate_pool(
    df,
    all_sequences,
    pool_size
):
    used_tracks = set()

    for seq in all_sequences:
        used_tracks.update(seq)

    available_tracks = df[
        ~df.index.isin(used_tracks)
    ]

    if len(available_tracks) < pool_size:
        return available_tracks.copy()

    return available_tracks.sample(
        pool_size,
        random_state=42
    ).copy()


def generate_playlist(
    model,
    df,
    all_sequences,
    sequence_idx,
    playlist_length,
    pool_size
):

    seed_sequence = all_sequences[
        sequence_idx
    ]

    input_track_ids = seed_sequence[:5]

    current_window = (
        df.loc[
            input_track_ids,
            MOOD_FEATURES
        ]
        .astype(float)
        .values
    )

    playlist_track_ids = (
        input_track_ids.copy()
    )

    candidate_pool = build_candidate_pool(
        df,
        all_sequences,
        pool_size
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

        next_track_id = next_track.name

        playlist_track_ids.append(
            next_track_id
        )

        candidate_pool = (
            candidate_pool.drop(
                next_track_id
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

    return df.loc[
        playlist_track_ids
    ]


# =====================================================
# LOAD DATA
# =====================================================

model = load_lstm_model()

df = load_tracks()

all_sequences = load_sequences()

# =====================================================
# UI
# =====================================================

st.title(
    "🎵 Genre-Aware Mood Journey Playlist Generator"
)

st.markdown(
    """
    Generate smooth listening journeys using
    an LSTM-based next-track prediction model.
    """
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Settings")

playlist_length = st.sidebar.slider(
    "Playlist Length",
    min_value=5,
    max_value=15,
    value=8
)

candidate_pool_size = st.sidebar.slider(
    "Candidate Pool Size",
    min_value=100,
    max_value=1000,
    value=300,
    step=50
)

sequence_idx = st.sidebar.selectbox(
    "Seed Sequence",
    range(len(all_sequences))
)

# =====================================================
# GENERATE BUTTON
# =====================================================

if st.button("Generate Playlist"):

    with st.spinner(
        "Generating mood journey..."
    ):

        playlist_df = generate_playlist(
            model,
            df,
            all_sequences,
            sequence_idx,
            playlist_length,
            candidate_pool_size
        )

    # =====================================
    # PLAYLIST TABLE
    # =====================================

    st.subheader(
        "Generated Playlist"
    )

    columns_to_show = []

    if "track_name" in playlist_df.columns:
        columns_to_show.append(
            "track_name"
        )

    if "artists" in playlist_df.columns:
        columns_to_show.append(
            "artists"
        )

    if "track_genre" in playlist_df.columns:
        columns_to_show.append(
            "track_genre"
        )

    if len(columns_to_show) > 0:

        st.dataframe(
            playlist_df[
                columns_to_show
            ],
            use_container_width=True
        )

    else:

        st.dataframe(
            playlist_df,
            use_container_width=True
        )

    # =====================================
    # SMOOTHNESS SCORE
    # =====================================

    generated_score = (
        smoothness_score(
            playlist_df[
                MOOD_FEATURES
            ].values
        )
    )

    random_playlist = df.sample(
        len(playlist_df),
        random_state=123
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

    # =====================================
    # MOOD JOURNEY
    # =====================================

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
            ].values
    })

    st.line_chart(
        chart_df.set_index(
            "Position"
        )
    )

    # =====================================
    # FEATURE TABLE
    # =====================================

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

    # =====================================
    # DOWNLOAD
    # =====================================

    csv_data = (
        playlist_df.to_csv(
            index=False
        )
    )

    st.download_button(
        label="Download Playlist CSV",
        data=csv_data,
        file_name="generated_playlist.csv",
        mime="text/csv"
    )


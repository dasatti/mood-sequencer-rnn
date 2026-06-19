import numpy as np
from scipy.spatial.distance import cdist


MOOD_FEATURES = [
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "tempo"
]


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
        ].values
    )

    distances = cdist(
        [predicted_vector],
        candidate_vectors,
        metric="euclidean"
    )[0]

    best_idx = np.argmin(distances)

    return candidate_pool.iloc[best_idx]


def smoothness_score(
    feature_matrix
):

    distances = []

    for i in range(
        len(feature_matrix) - 1
    ):

        d = np.linalg.norm(
            feature_matrix[i + 1]
            -
            feature_matrix[i]
        )

        distances.append(d)

    return np.mean(distances)
from pykalman import KalmanFilter


def apply_kalman(series):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        observation_covariance=1,
        transition_covariance=0.01
    )

    state_means, _ = kf.filter(series.values)

    return state_means.flatten()

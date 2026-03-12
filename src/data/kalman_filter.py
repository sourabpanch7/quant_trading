from pykalman import KalmanFilter


def apply_kalman(price_series):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=price_series.iloc[0],
        observation_covariance=1,
        transition_covariance=0.01
    )

    state_means, _ = kf.filter(price_series.values)

    return state_means.flatten()

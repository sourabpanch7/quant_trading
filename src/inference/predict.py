import numpy as np


def generate_signals(predictions, threshold=0.002):
    signals = []

    for p in predictions:

        if p > threshold:
            signals.append(1)

        elif p < -threshold:
            signals.append(-1)

        else:
            signals.append(0)

    return np.array(signals)

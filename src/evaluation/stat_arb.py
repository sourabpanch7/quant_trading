from statsmodels.tsa.stattools import coint


def find_pairs(price_df):
    n = price_df.shape[1]
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            score, pvalue, _ = coint(price_df.iloc[:, i],
                                     price_df.iloc[:, j])
            if pvalue < 0.05:
                pairs.append((i, j, pvalue))

    return pairs


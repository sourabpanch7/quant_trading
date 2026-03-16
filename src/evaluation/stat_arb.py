import numpy as np


def find_stat_arb_pairs(df, corr_threshold=0.15):
    returns_pivot = df.pivot_table(
        index="Date",
        columns="stock_id",
        values="actual_return"
    )

    corr_matrix = returns_pivot.corr()

    pairs = []

    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i < j and np.abs(corr_matrix.loc[i, j]) > corr_threshold:
                pairs.append((i, j, corr_matrix.loc[i, j]))

    return pairs


def run_stat_arb_strategy(spread_df, entry_threshold=1, exit_threshold=0.5):
    spread_df["position"] = 0
    spread_df.loc[spread_df["zscore"] > entry_threshold, "position"] = -1
    spread_df.loc[spread_df["zscore"] < -entry_threshold, "position"] = 1
    spread_df["position"] = spread_df["position"].replace(0, np.nan).ffill().fillna(0)
    spread_df.loc[np.abs(spread_df["zscore"]) < exit_threshold, "position"] = 0
    spread_df["spread_return"] = spread_df["spread"].diff()
    spread_df["strategy_return"] = spread_df["position"].shift(1) * spread_df["spread_return"]

    return spread_df

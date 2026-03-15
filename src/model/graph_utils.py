import torch


def build_stock_graph(df, threshold=0.2):
    pivot = df.pivot_table(
        index="Date",
        columns="stock_id",
        values="log_return"
    )

    corr = pivot.corr()

    edges = []

    for i in range(len(corr)):
        for j in range(len(corr)):
            if i != j and abs(corr.iloc[i, j]) > threshold:
                edges.append([i, j])

    edge_index = torch.tensor(edges).t().contiguous()

    return edge_index

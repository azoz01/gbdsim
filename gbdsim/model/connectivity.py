from torch import Tensor, histc, stack, zeros


def mutual_information(x: Tensor, y, num_bins=30):
    joint_hist = histc(stack([x, y], dim=1), bins=num_bins)
    px = histc(x, bins=num_bins) / x.numel() + 1e-15
    py = histc(y, bins=num_bins) / y.numel() + 1e-15

    pxy = joint_hist / joint_hist.sum()
    px_py = px.unsqueeze(1) * py.unsqueeze(0)

    mi = (pxy * (pxy / px_py).log()).nansum()
    return mi.item()


def pairwise_mutual_information(x: Tensor, num_bins=30):
    marginal_densities = [
        histc(col, bins=num_bins) / col.numel() for col in x.T
    ]
    mi = zeros(x.shape[1], x.shape[1])
    for i in range(x.shape[1]):
        for j in range(i + 1):
            c1 = x[:, i]
            c2 = x[:, j]
            cell = histc(stack([c1, c2], dim=1), bins=num_bins)
            cell /= cell.sum()
            denom = (
                marginal_densities[i].unsqueeze(1)
                * marginal_densities[j].unsqueeze(0)
                + 1e-15
            )
            mi[i][j] = (cell * (cell / denom).log()).nansum()
            if i == j:
                mi[i][j] /= 2
    return mi + mi.T

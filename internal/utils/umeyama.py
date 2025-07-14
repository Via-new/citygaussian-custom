import torch

def umeyama_torch(X: torch.Tensor, Y: torch.Tensor):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : torch.Tensor
        (m, n) shaped tensor. m is the dimension of the points,
        n is the number of points in the point set.
    Y : torch.Tensor
        (m, n) shaped tensor. Indexes should be consistent with `X`.

    Returns
    -------
    c : float
        Scale factor.
    R : torch.Tensor
        (3, 3) shaped rotation matrix.
    t : torch.Tensor
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(dim=1, keepdim=True)
    mu_y = Y.mean(dim=1, keepdim=True)
    var_x = ((X - mu_x) ** 2).sum(dim=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]

    U, D, Vh = torch.linalg.svd(cov_xy, full_matrices=True)

    S = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    if torch.det(U) * torch.det(Vh) < 0:
        S[-1, -1] = -1

    c = torch.trace(torch.diag(D) @ S) / var_x
    R = U @ S @ Vh
    t = mu_y - c * R @ mu_x

    return c, R, t

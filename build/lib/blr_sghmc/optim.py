import numpy as np
from numpy.random import multivariate_normal


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prior(params):
    a, b = 1, 2
    return a / (2 * ((b ** 2) ** 0.5)) * np.exp(-a * np.abs(params) / ((b ** 2) ** 0.5))


def del_prior(params):
    a, b = 1, 2
    output = -(
        a ** 2
        * ((params ** 2) ** 0.5)
        * np.exp(-a * ((params ** 2) ** 0.5) / ((b ** 2) ** 0.5))
    ) / (2 * b ** 2 * params)
    return output


def H(X, y, params, rs, M_inv):
    return U(X, y, params) + K(rs, M_inv)


def K(rs, M_inv):
    return 1 / 2 * np.sum(rs @ M_inv @ rs.T)


def U(X, y, params):
    return -(
        np.log(sigmoid(X[y == 1] @ params)).sum()
        + np.log(1 - sigmoid(X[y == 0] @ params)).sum()
        + np.log(prior(params)).sum()
    )


def del_U(X, y, params, scale=1):
    return -(
        scale * ((y[:, None] - sigmoid(X @ params)).T @ X).flatten()
        + del_prior(params).flatten()
    )


def V(X, y, params):
    return np.cov((((y[:, None] - sigmoid(X @ params)).T) * X.T))


def hmc(X, y, M, penalty="l1", max_iter=500, m=1000, eps=1e-5, verbose=False):
    n_params = X.shape[1]
    params_t = np.random.normal(size=(n_params, 1))
    accepted = []
    M_inv = np.linalg.inv(M)
    rs = multivariate_normal(np.zeros(n_params), M, size=(max_iter, 1))
    us = np.random.uniform(size=(max_iter))
    for t in range(max_iter):
        params_i, r_i = params_t.copy(), rs[t].copy()
        r_i = r_i - eps / 2 * del_U(X, y, params_i)
        for _ in range(m):
            params_i += eps * (M_inv @ r_i.T)
            r_i -= eps * del_U(X, y, params_i)
        r_i = r_i - eps / 2 * del_U(X, y, params_i)

        # MH correction
        rho = np.exp(H(X, y, params_i, r_i, M_inv) - H(X, y, params_t, rs[t], M_inv))
        if us[t] < min(1, rho):
            params_t = params_i
            accepted.append(params_t)

        if verbose:
            if not t % 100:
                print(
                    f"Iteration {t:>5}"
                    + "".join(f"| {p:>7.4f}" for p in params_t.flatten())
                )
    est_params = np.r_[accepted][int(max_iter * 0.7) :].mean(axis=0)

    return est_params


def sghmc(
    X, y, M, batch_size=16, penalty="l1", max_iter=500, m=1000, eps=1e-5, verbose=False
):
    n_samples, n_params = X.shape[0], X.shape[1]
    scale = n_samples / batch_size
    params = np.random.normal(size=n_params).reshape(-1, 1)
    M_inv = np.linalg.inv(M)
    rs = multivariate_normal(np.zeros(n_params), M, size=(max_iter, 1))
    batch_ids = np.random.choice(range(n_samples), size=(max_iter, batch_size))
    for t in range(max_iter):
        batch_X, batch_y = X[batch_ids[t]], y[batch_ids[t]]
        for _ in range(m):
            params += eps * (M_inv @ rs[t].T)
            B_hat = 1 / 2 * eps * V(batch_X, batch_y, params)
            C = B_hat
            if eps > 1e-3:
                rs[t] -= (
                    eps * del_U(batch_X, batch_y, params, scale=scale)
                    + eps * (C @ (M_inv @ rs[t].T)).flatten()
                    - multivariate_normal(np.zeros(n_params), 2 * (C - B_hat) * eps)
                )
            else:
                rs[t] -= (
                    eps * del_U(batch_X, batch_y, params, scale=n_samples / batch_size)
                    + eps * (C @ (M_inv @ rs[t].T)).flatten()
                )
        if verbose:
            if not t % 100:
                print(
                    f"Iteration {t:>5}"
                    + "".join(f"| {p:>7.4f}" for p in params.flatten())
                )

    return params

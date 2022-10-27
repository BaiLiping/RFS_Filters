import numpy as np


def gauss(x: np.array, mu: np.array, P: np.array) -> np.float:
    n = mu.shape[1]
    Pinv = np.linalg.inv(P)
    res = ((1/(np.sqrt((2*np.pi))**n)) * np.exp((-1 / 2) * (x - mu).T@Pinv@(x - mu)))[0][0]
    return res

__all__ = ["normalize_by_max", "dot_product"]

from typing import Literal
import numpy as np


def normalize_by_max(x: np.ndarray) -> np.ndarray:
    x_max = np.nanmax(np.abs(x))
    if x_max == 0.0:
        return x
    else:
        return np.divide(x, x_max)


def dot_product(x: np.ndarray, y: np.ndarray):
    # avoid overflow
    x = normalize_by_max(x)
    y = normalize_by_max(y)
    prod1 = np.nansum(np.multiply(x, x))
    prod2 = np.nansum(np.multiply(y, y))
    if prod1 == 0.0 or prod2 == 0.0:
        return np.float64(0.0)
    return np.multiply(x, y).sum() / np.sqrt(np.multiply(prod1, prod2))


def spectral_angle(x: np.ndarray, y: np.ndarray):
    dot_prod = dot_product(x, y)
    return np.arccos(dot_prod) / np.pi * 2


def jaccard(x: np.ndarray, y: np.ndarray, cutoff: float = 0.01):
    x = normalize_by_max(x)
    y = normalize_by_max(y)

    intersect = np.nansum((x > cutoff) & (y > cutoff))
    union = np.nansum((x > cutoff) | (y > cutoff))
    if union == 0:
        return np.float64(0.0)
    return intersect / union


def optional_jaccard(x: np.ndarray, y: np.ndarray, y_essential: np.ndarray, cutoff: float = 0.01):
    x = normalize_by_max(x)
    y = normalize_by_max(y)

    x = x > cutoff
    y = y > cutoff

    if np.nansum(x & y & y_essential.astype(bool)) == 0:
        return np.float64(0.0)

    y = np.where(~x & ~y_essential.astype(bool), False, y)

    intersect = np.nansum(x & y)
    union = np.nansum(x | y)

    if union == 0:
        return np.float64(0.0)
    return intersect / union


def weighted_jaccard(x: np.ndarray, y: np.ndarray):
    x = normalize_by_max(x)
    y = normalize_by_max(y)

    intersect = np.nansum(np.minimum(x, y))
    union = np.nansum(np.maximum(x, y))
    if union == 0:
        return np.float64(0.0)
    return intersect / union

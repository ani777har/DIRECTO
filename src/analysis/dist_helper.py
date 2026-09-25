###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
# import pyemd
import hashlib
import numpy as np
import concurrent.futures
from collections import OrderedDict
from functools import partial
from scipy.linalg import toeplitz
from tqdm import tqdm


def l2(x, y):
    dist = np.linalg.norm(x - y, 2)
    return dist


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
        x, y: 1D pmf of two distributions with the same support
        sigma: standard deviation
    """
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = compute_emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x, y, sigma=1.0):
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """Discrepancy between 2 samples"""
    d = 0

    if not is_parallel:
        for s1 in tqdm(samples1):
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dist in tqdm(
                executor.map(
                    kernel_parallel_worker,
                    [
                        (s1, samples2, partial(kernel, *args, **kwargs))
                        for s1 in samples1
                    ],
                ),
                total=len(samples1),
            ):
                d += dist
    if len(samples1) * len(samples2) > 0:
        d /= len(samples1) * len(samples2)
    else:
        d = 1e6
    return d


# Cache for the MMD reference self-term, which is constant across search trials.
_SELF_DISC_CACHE = OrderedDict()
_SELF_DISC_CACHE_MAXSIZE = 64
_SELF_DISC_CACHE_ENABLED = False
_SELF_DISC_CACHE_MIN_SAMPLES = 64


def enable_self_disc_cache(enabled=True):
    global _SELF_DISC_CACHE_ENABLED
    _SELF_DISC_CACHE_ENABLED = enabled
    _SELF_DISC_CACHE.clear()


def clear_self_disc_cache():
    _SELF_DISC_CACHE.clear()


def _samples_digest(samples):
    h = hashlib.blake2b(digest_size=16)
    h.update(str(len(samples)).encode())
    for s in samples:
        a = np.ascontiguousarray(s)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.digest()


def self_disc(samples, kernel, *args, **kwargs):
    """disc(samples, samples), memoized on the content of ``samples``."""
    if not _SELF_DISC_CACHE_ENABLED or len(samples) < _SELF_DISC_CACHE_MIN_SAMPLES:
        return disc(samples, samples, kernel, *args, **kwargs)

    key = (
        getattr(kernel, "__module__", None),
        getattr(kernel, "__qualname__", repr(kernel)),
        args,
        tuple(sorted(kwargs.items())),
        _samples_digest(samples),
    )
    if key in _SELF_DISC_CACHE:
        _SELF_DISC_CACHE.move_to_end(key)
        return _SELF_DISC_CACHE[key]

    value = disc(samples, samples, kernel, *args, **kwargs)
    _SELF_DISC_CACHE[key] = value
    if len(_SELF_DISC_CACHE) > _SELF_DISC_CACHE_MAXSIZE:
        _SELF_DISC_CACHE.popitem(last=False)
    return value


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """MMD between two samples"""
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / (np.sum(s1) + 1e-6) for s1 in samples1]
        samples2 = [s2 / (np.sum(s2) + 1e-6) for s2 in samples2]
    return (
        self_disc(samples1, kernel, *args, **kwargs)
        + self_disc(samples2, kernel, *args, **kwargs)
        - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
    )


def compute_emd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """EMD between average of two samples"""
    # normalize histograms into pmf
    if is_hist:
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    return disc(samples1, samples2, kernel, *args, **kwargs), [samples1[0], samples2[0]]

import numpy as np
import math
from scipy.signal import savgol_filter


def _reject_outliers_1d(signal, window=5, threshold=3.0):
    """Sliding-window MAD outlier detection and linear interpolation for 1D signal.
    Returns (cleaned, is_outlier)."""
    T = len(signal)
    half = window // 2
    is_outlier = np.zeros(T, dtype=bool)

    global_med = np.median(signal)
    global_mad = np.median(np.abs(signal - global_med))

    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        w = signal[lo:hi]
        med = np.median(w)
        local_mad = np.median(np.abs(w - med))
        scale = local_mad if local_mad > 1e-8 else global_mad
        if scale > 1e-8 and np.abs(signal[t] - med) > threshold * scale:
            is_outlier[t] = True

    if not np.any(is_outlier):
        return signal.copy(), is_outlier

    valid = np.where(~is_outlier)[0]
    cleaned = signal.copy()
    if len(valid) >= 2:
        cleaned = np.interp(np.arange(T), valid, signal[valid])
    elif len(valid) == 1:
        cleaned[:] = signal[valid[0]]
    return cleaned, is_outlier


def reject_outliers_median_2d(signal, window=5, threshold=3.0):
    """Outlier rejection for (T, N) signal, applied independently per dimension."""
    T, N = signal.shape
    cleaned = np.empty_like(signal)
    any_outlier = np.zeros(T, dtype=bool)
    for n in range(N):
        cleaned[:, n], is_outlier = _reject_outliers_1d(signal[:, n], window, threshold)
        any_outlier |= is_outlier
    print(f"  [outlier rejection 2d] {np.sum(any_outlier)}/{T} frames rejected")
    return cleaned


def reject_outliers_median_3d(data, window=5, threshold=3.0):
    """Outlier rejection for (T, J, N) data.

    Per joint: frames where the position deviates from the window median by more
    than ``threshold`` * MAD (in Euclidean distance) are flagged as outliers and
    replaced by linear interpolation of the nearest valid neighbours.
    """
    T, J, N = data.shape
    half = window // 2
    cleaned = data.copy()

    any_outlier = np.zeros(T, dtype=bool)
    for j in range(J):
        traj = data[:, j, :]  # (T, N)
        is_outlier = np.zeros(T, dtype=bool)

        global_med = np.median(traj, axis=0)
        global_mad = np.median(np.linalg.norm(traj - global_med, axis=1))

        for t in range(T):
            lo = max(0, t - half)
            hi = min(T, t + half + 1)
            w = traj[lo:hi]  # (w_size, N)
            med = np.median(w, axis=0)  # (N,)
            dists = np.linalg.norm(w - med, axis=1)  # (w_size,)
            local_mad = np.median(dists)
            scale = local_mad if local_mad > 1e-8 else global_mad
            if scale > 1e-8 and np.linalg.norm(traj[t] - med) > threshold * scale:
                is_outlier[t] = True

        any_outlier |= is_outlier
        if not np.any(is_outlier):
            continue

        valid = np.where(~is_outlier)[0]
        if len(valid) >= 2:
            all_t = np.arange(T)
            for n in range(N):
                cleaned[:, j, n] = np.interp(all_t, valid, traj[valid, n])
        elif len(valid) == 1:
            cleaned[:, j, :] = traj[valid[0]]

    print(f"  [outlier rejection 3d] {np.sum(any_outlier)}/{T} frames rejected")
    return cleaned


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

def apply_one_euro_filter_2d(signal, mincutoff = 1.0, beta = 0.0, dcutoff = 1.0):
    T, N = signal.shape
    times = np.linspace(0, T-1, T)
    poses_filter = OneEuroFilter(times[0], signal[0], np.zeros(N), min_cutoff = mincutoff, beta = beta, d_cutoff = dcutoff)
    filtered_signal = np.array([poses_filter(np.asarray([times[i]]), signal[i]) for i in range(T)])
    return filtered_signal.squeeze(1)

def _clamp_savgol_params(T, window, polyorder):
    if window % 2 == 0:
        window += 1
    window = min(window, T if T % 2 == 1 else T - 1)
    polyorder = min(polyorder, window - 1)
    return window, polyorder


def apply_savgol_filter_2d(signal, window=11, polyorder=3):
    """Zero-phase Savitzky-Golay smoothing for (T, N) signal."""
    T, N = signal.shape
    window, polyorder = _clamp_savgol_params(T, window, polyorder)
    filtered = signal.copy()
    for n in range(N):
        filtered[:, n] = savgol_filter(signal[:, n], window_length=window, polyorder=polyorder)
    return filtered


def apply_savgol_filter_3d(data, window=11, polyorder=3):
    """Zero-phase Savitzky-Golay smoothing for (T, J, N) data.

    Unlike the causal One Euro filter, this is a symmetric filter so it
    introduces no phase lag / drift.  ``window`` must be odd and greater than
    ``polyorder``.
    """
    T, J, N = data.shape
    window, polyorder = _clamp_savgol_params(T, window, polyorder)
    filtered = data.copy()
    for j in range(J):
        for n in range(N):
            filtered[:, j, n] = savgol_filter(data[:, j, n], window_length=window, polyorder=polyorder)
    return filtered


def apply_one_euro_filter_3d(data, mincutoff = 1.0, beta = 0.0, dcutoff = 1.0):
    T, J, N = data.shape

    times = np.linspace(0, T-1, T)
    filters = [OneEuroFilter(times[0], data[0, j], np.zeros(N), min_cutoff = mincutoff, beta = beta, d_cutoff = dcutoff) for j in range(J)]

    filtered_data = np.zeros_like(data)
    for t in range(T):
        for j in range(J):
            filtered_data[t, j] = filters[j](np.asarray([times[t]]), data[t, j])
    
    return filtered_data
            
class OneEuroFilter:
    def __init__(self, t0, x0, dx0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0.copy()
        self.dx_prev = dx0.copy()
        self.t_prev = t0.copy()

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e[:,np.newaxis]
        dx[~np.isfinite(dx)] = 0
        dx_hat = exponential_smoothing(a_d[:,np.newaxis], dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e[:,np.newaxis], cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t.copy()

        return x_hat
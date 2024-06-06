import numpy as np
import math

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
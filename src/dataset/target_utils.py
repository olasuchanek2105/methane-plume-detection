import numpy as np 

def normalize_methane_rate(values):
    values = np.array(values, dtype=np.float32)

    min_v = np.min(values)
    max_v = np.max(values)


    if max_v - min_v < 1e-8:
        values_norm = np.zeros(len(values))
    else: values_norm = (values - min_v)/(max_v - min_v)

    return values_norm


def log_minmax_normalize(values):
    values = np.array(values)
    log_values = np.log1p(values)
    normalized = normalize_methane_rate(log_values)
    return normalized
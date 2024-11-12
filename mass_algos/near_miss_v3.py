import numpy as np
from fastdtw import fastdtw

"""
V3 Update: Near Miss Algorithm fixed to z-normalize the sequences before computing the DTW distance. Additionally the DTW distance
is normalized by the warping path length to ensure that the distance is not biased by the length of the sequences.

This module implements a Dynamic Time Warping (DTW)-based near-miss detection algorithm to identify segments in long time series 
data that closely resemble a given query pattern, with an additional scale penalty. 

The primary function, `near_miss`, computes a distance profile by sliding a query sequence (representing a "near miss" pattern) 
over a larger time series (representing trip data) and calculating the DTW distance between each segment and the query. To adjust
 for differences in scale, a custom scale penalty is applied based on the deviation in means between each segment and the query.

Functions included:
- `near_miss`: Main function to generate a DTW-based distance profile with scale penalty.
- `compute_scale_penalty`: Calculates a scale penalty for segments, based on mean differences.
- `determine_k`: Determines the optimal segment size for efficiency, balancing accuracy and computational load.

Usage:
This implementation is particularly useful in identifying near-miss patterns within continuous data streams, such as vehicle 
dynamics data, where a precise match of a known risky sequence can highlight potentially hazardous behavior.

Dependencies:
- `numpy` for numerical operations
- `fastdtw` for fast, approximate DTW calculations
"""


def near_miss(x, y, r=3, scale_weight=0):
    """
    Parameters:
    x : numpy array
        Long time series data (Trip)
    y : numpy array
        Query sequence (Near Miss Pattern)
    r : int
        Size of neighborhood when expanding the path. A higher value will
        increase the accuracy of the calculation but also increase time
        and memory consumption. A radius equal to the size of x and y will
        yield an exact dynamic time warping calculation.
    Returns:
    dist : numpy array
        DTW-based distance profile with scale penalty
    """

    n = len(x)
    m = len(y)
    half_m = m // 2  # Half the length of the query
    expected_length = n - m + 1
    dist = []

    k = determine_k(n, m)

    y_norm = z_normalize(y)

    # Loop through the time series x with varying segment lengths
    for j in range(0, n - m + 1, k):
        segment_lengths = range(m - half_m, m + half_m + 1)
        best_distance = float('inf')

        for seg_len in segment_lengths:
            if j + seg_len <= n:
                segment = x[j:j + seg_len]
                segment_norm = z_normalize(segment)

                # Compute DTW distance and path
                distance, path = fastdtw(segment_norm, y_norm, r)

                # Normalize the distance by warping path length
                path_length = len(path)
                normalized_distance = distance / path_length

                # Compute scale penalty if sequences are not normalized
                if scale_weight > 0:
                    scale_penalty = compute_scale_penalty(segment, y)
                else:
                    scale_penalty = 0
                total_distance = normalized_distance + scale_weight * scale_penalty

                # Keep the best distance for this starting point
                if total_distance < best_distance:
                    best_distance = total_distance

        # Append the best distance found for this position
        dist.append(best_distance)

    # Ensure consistent length
    dist = np.array(dist)
    if len(dist) > expected_length:
        dist = dist[:expected_length]
    elif len(dist) < expected_length:
        dist = np.pad(dist, (0, expected_length - len(dist)), 'constant', constant_values=np.nan)

    return dist

def z_normalize(ts):
    """
    Z-normalizes a time series with protection against edge cases.
    """
    std = np.std(ts)
    if std == 0:  # Handle constant sequences
        return np.zeros_like(ts)
    if np.any(np.isnan(ts)) or np.any(np.isinf(ts)):  # Handle invalid values
        return np.zeros_like(ts)
    return (ts - np.mean(ts)) / std

def compute_scale_penalty(subsequence, query_sequence):
    """
    Computes the scale penalty based on the difference in scale (mean and std deviation)
    between a subsequence and the query sequence.

    Parameters:
    ----------
    subsequence : numpy array
        A subsequence of the longer time series.
    query_sequence : numpy array
        The query time series.

    Returns:
    -------
    scale_penalty : float
        A penalty value based on the scale difference between the two sequences.
    """
    # Calculate the length of the subsequence
    len_subsequence = len(subsequence)

    # Calculate mean and standard deviation of both sequences
    mean_subsequence = np.mean(subsequence)
    mean_query = np.mean(query_sequence)
    
    # Compute penalty as the absolute difference in means
    mean_diff = abs(mean_subsequence - mean_query)
    
    scale_penalty = len_subsequence * mean_diff # Sequence length multiplied by mean difference
    
    return scale_penalty

def determine_k(n, m):
    """
    Determines the optimal value of k.
    
    Parameters:
    n : Length of the time series (int) 
    m : Length of the query (int)

    Returns:
    k : Optimal segment size, preferably a power of two (int)
    """
    # Set k to be the next power of two greater than or equal to 4 times the query length
    k = 2 ** int(np.ceil(np.log2(max(4 * m, m))))
    # Ensure k is not greater than the length of the time series
    k = min(k, n)
    return k

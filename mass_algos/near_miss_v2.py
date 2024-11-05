import numpy as np
from fastdtw import fastdtw

"""
V2 Update: Near Miss Algorithm updated to compare the query with varying length sub-sequences of the longer time series to 
account for the different rate of actions

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


def near_miss(x, y, r=3):
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
    half_m = m // 2 # half the length of the query and round down
    expected_length = n - m + 1
    dist = []

    k = determine_k(n, m) # Size of pieces (Refer to MASS V3)

    # Loop through the time series x with varying segment lengths
    for j in range(0, n - m + 1, k):
        segment_lengths = list(range(m - half_m, m + half_m + 1)) # Test with different segment lengths to find better matches of varying lengths
        best_distance = float('inf')

        for seg_len in segment_lengths:
            if j + seg_len <= n:  # Ensure segment doesn't exceed bounds
                segment = x[j:j + seg_len]

                # Use DTW to compute distance between the current segment and the query
                distance, _ = fastdtw(segment, y)
                distance += compute_scale_penalty(segment, y)

                # Keep the best distance for this starting point
                if distance < best_distance:
                    best_distance = distance

        # Append the best distance found for this position
        dist.append(best_distance)

    # Ensure consistent length
    dist = np.array(dist)
    if len(dist) > expected_length:
        dist = dist[:expected_length]
    elif len(dist) < expected_length:
        dist = np.pad(dist, (0, expected_length - len(dist)), 'constant', constant_values=np.nan)

    return dist

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

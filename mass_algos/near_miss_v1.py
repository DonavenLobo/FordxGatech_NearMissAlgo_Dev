import numpy as np
from fastdtw import fastdtw

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
    expected_length = n - m + 1
    dist = []

    k = determine_k(n, m) # Size of pieces (Refer to MASS V3)

    # Loop through segments of the time series x
    for j in range(0, n - m + 1, k):  # Adjust step size based on query length and window
        segment = x[j:j + m]
        
        if len(segment) == m:  # Ensure segment matches query length
            # Use DTW to compute distance between the current segment and the query
            distance, _ = fastdtw(segment, y)
            distance += compute_scale_penalty(segment, y)
            dist.append(distance)

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
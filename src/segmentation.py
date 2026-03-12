
import numpy as np

def segment_variance(x):
    return np.var(x)

def segment_recursive(x, left, right, threshold, segments):
    seg = x[left:right]
    if len(seg) <= 32 or segment_variance(seg) <= threshold:
        segments.append((left, right))
        return
    
    mid = (left + right) // 2
    segment_recursive(x, left, mid, threshold, segments)
    segment_recursive(x, mid, right, threshold, segments)

def run_segmentation(signal, threshold):
    segments = []
    segment_recursive(signal, 0, len(signal), threshold, segments)
    return segments

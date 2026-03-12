import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from segmentation import run_segmentation
from clustering import top_down_cluster
from kadane import kadane

# Convert RUL value into a class
def rul_class(x):
    if x < Q10: return "Extremely Low"
    elif x < Q40: return "Moderately Low"
    elif x < Q90: return "Moderately High"
    else: return "Extremely High"

# Get majority class from a list
def majority_class(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get), counts[max(counts, key=counts.get)]

# Plot segmentation boundaries for a sensor
def plot_segments(signal, segments, sensor_name):
    plt.figure(figsize=(12, 4))
    plt.plot(signal, linewidth=1)
    for left, right in segments:
        plt.axvline(left, linestyle="--", linewidth=0.7)
        plt.axvline(right - 1, linestyle="--", linewidth=0.7)
    plt.title(f"{sensor_name} Segmentation")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

# Load dataset and select sensor columns
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "rul_hrs.csv"))
sensor_cols = [c for c in df.columns if "sensor" in c]
df = df.iloc[:10000].copy()

# Compute RUL quantiles and create class labels
Q10 = df["rul"].quantile(.1)
Q40 = df["rul"].quantile(.4)
Q90 = df["rul"].quantile(.9)
df["class"] = df["rul"].apply(rul_class)

# Select 10 sensors for segmentation
random.seed(42)
selected_sensors = random.sample(sensor_cols, 10)

# Task 1: Segmentation
print("\n--- Task 1: Segmentation ---")
segmentation_scores = []

for sensor in selected_sensors:
    signal = df[sensor].values
    threshold = np.var(signal) / 2
    segments = run_segmentation(signal, threshold)
    complexity_score = len(segments)
    segmentation_scores.append((sensor, complexity_score))

    segment_classes = []
    for left, right in segments:
        seg_labels = df["class"].iloc[left:right].tolist()
        dominant, count = majority_class(seg_labels)
        segment_classes.append(dominant)

    print(sensor, "| Segments:", complexity_score, "| Dominant segment classes:", segment_classes[:5])
    plot_segments(signal, segments, sensor)

print("\nSegmentation Complexity Scores")
for sensor, score in segmentation_scores:
    print(sensor, "->", score)

# Task 2: Clustering
print("\n--- Task 2: Clustering ---")
X = df[sensor_cols].values
clusters = top_down_cluster(X, 4)
print("Clusters created:", len(clusters))

for i, (cluster_data, cluster_indices) in enumerate(clusters):
    cluster_labels = df.iloc[cluster_indices]["class"].tolist()
    dominant, count = majority_class(cluster_labels)

    print("Cluster", i, "| Size:", len(cluster_indices),
          "| Majority Class:", dominant, "| Count:", count)

# Task 3: Kadane analysis
print("\n--- Task 3: Kadane Analysis ---")
kadane_results = []

for sensor in sensor_cols:
    signal = df[sensor].values

    # Compute absolute first difference and center it
    d = np.abs(np.diff(signal))
    x = d - np.mean(d)

    # Find max deviation interval
    start, end, score = kadane(x)

    # Match interval to RUL classes
    interval_classes = df["class"].iloc[start:end+1].tolist()
    dominant, count = majority_class(interval_classes)

    kadane_results.append((sensor, start, end, score, dominant, count))

    print(sensor, "| Interval:", (start, end),
          "| Score:", round(score, 3),
          "| Dominant Class:", dominant)

print("\nSensors whose max deviation falls in low RUL")
for sensor, start, end, score, dominant, count in kadane_results:
    if dominant in ["Extremely Low", "Moderately Low"]:
        print(sensor, "| Interval:", (start, end),
              "| Score:", round(score, 3),
              "| Class:", dominant)
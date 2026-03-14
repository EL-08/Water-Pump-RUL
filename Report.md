
# Algorithmic Time‑Series Segmentation and Condition Analysis

## Project Overview

This project analyzes time‑series sensor data from a water‑pump predictive maintenance dataset using core algorithms instead of machine learning.  
The goal is to study how sensor behavior relates to machine health states derived from Remaining Useful Life (RUL).

Three algorithms were implemented:
- Divide‑and‑Conquer Segmentation
- Divide‑and‑Conquer Clustering
- Kadane’s Maximum Subarray

These algorithms analyze signal structure, group machine states, and detect sustained deviations in sensor readings.

---

## Installation and Usage Instructions

### Requirements

Python packages:
- pandas  
- numpy  
- matplotlib  

Install:

```
pip install -r requirements.txt
```

### Run the project

From the project root:

```
python src/main.py
```

The script will:
1. Load the dataset
2. Run segmentation on selected sensors
3. Cluster machine states
4. Run Kadane analysis
5. Print results and show segmentation plots

---

## Code Structure

### main.py
Controls the whole pipeline:
- loads dataset
- creates RUL categories
- runs segmentation
- runs clustering
- runs Kadane analysis

### segmentation.py
Implements recursive Divide‑and‑Conquer segmentation based on variance.

### clustering.py
Implements a simple top‑down clustering algorithm.

### kadane.py
Implements Kadane’s algorithm to find the interval with the largest cumulative deviation.

---

## Algorithm Descriptions

### Divide‑and‑Conquer Segmentation

A signal is recursively split until the variance of each segment falls below a threshold.

Steps:

1. Compute variance
2. If variance > threshold → split
3. Recursively repeat

The final number of segments is the **Segmentation Complexity Score**.

Higher score → more signal variability.

---

### Divide‑and‑Conquer Clustering

The dataset is recursively partitioned until 4 clusters are created.

Steps:

1. Start with all points in one cluster
2. Find feature with highest variance
3. Split cluster along that dimension
4. Repeat until 4 clusters exist

Clusters are then compared with the true RUL classes.

---

### Kadane Maximum Subarray

Kadane finds the contiguous interval with the largest sum.

Signal preprocessing:

```
d[i] = |sensor[i] − sensor[i−1]|
x[i] = d[i] − mean(d)
```

Kadane then identifies the interval with the strongest sustained deviation.

---

## Toy Example Verification

### Segmentation

Signal:

```
[2,2,2,9,9,9]
```

Segments:

```
[2,2,2] | [9,9,9]
```

---

### Clustering

Dataset:

```
[[1,1],[1,2],[9,9],[8,9]]
```

Clusters:

```
Cluster 1: [1,1],[1,2]
Cluster 2: [9,9],[8,9]
```

---

### Kadane

Array:

```
[-2,3,5,-1,4,-6]
```

Maximum subarray:

```
[3,5,-1,4]
```

Sum = 11

---

## Dataset Description

Dataset: Water Pump RUL – Predictive Maintenance
The analysis uses the first 10,000 rows.
Each row represents one timestamp.

Columns include:
- timestamp
- sensor_00 – sensor_51
- rul (Remaining Useful Life)

---

### Variables Used

- 52 sensor signals
- RUL value
- derived RUL class

---

### RUL → 4 Category Transformation

RUL values are converted using quantiles.

```
Q10 = 10th percentile
Q40 = 40th percentile
Q90 = 90th percentile
```

| Condition | Definition |
|----------|-----------|
| Extremely Low | RUL < Q10 |
| Moderately Low | Q10 ≤ RUL < Q40 |
| Moderately High | Q40 ≤ RUL < Q90 |
| Extremely High | RUL ≥ Q90 |

---

## Execution Results

### Task 1 — Segmentation

Segmentation was applied to 10 sensors.
Each sensor produces a Segmentation Complexity Score.
Higher scores indicate more signal changes.

---

### Task 2 — Clustering

All 10,000 rows were grouped into 4 clusters.
For each cluster the majority RUL class was computed.
Clusters dominated by a class suggest a relationship between sensor patterns and machine health.

---

### Task 3 — Kadane Analysis

Kadane was applied to all sensors.
For each sensor the algorithm returns:
- start index
- end index
- deviation score

The dominant RUL class inside the interval was also recorded.
Sensors whose intervals occur during low‑RUL periods may indicate degradation.

---

## Discussion and Conclusions

### Findings

Some sensors show increased variation as RUL decreases.  
Segmentation complexity often increases near failure states.
Clustering groups machine states that frequently align with RUL categories.
Kadane analysis highlights sensors with strong deviations during low‑RUL intervals.

---

### Challenges

- handling high‑dimensional sensor data
- implementing clustering without ML libraries
- selecting segmentation thresholds

---

### Limitations and Improvements

Limitations:
- only 10k rows analyzed

Possible improvements:
- analyze full dataset
- test alternative clustering strategies

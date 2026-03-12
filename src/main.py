
import pandas as pd
import numpy as np
from segmentation import run_segmentation
from clustering import top_down_cluster
from kadane import kadane

df = pd.read_csv("../data/rul_hrs.csv")

sensor_cols = [c for c in df.columns if "sensor" in c]

df = df.iloc[:10000]

Q10 = df['rul'].quantile(.1)
Q40 = df['rul'].quantile(.4)
Q90 = df['rul'].quantile(.9)

def rul_class(x):
    if x < Q10: return "Extremely Low"
    elif x < Q40: return "Moderately Low"
    elif x < Q90: return "Moderately High"
    else: return "Extremely High"

df['class'] = df['rul'].apply(rul_class)

sensor = sensor_cols[0]
signal = df[sensor].values

threshold = np.var(signal)/2

segments = run_segmentation(signal,threshold)

print("Segmentation segments:",len(segments))

X = df[sensor_cols].values
clusters = top_down_cluster(X,4)

print("Clusters created:",len(clusters))

d = np.abs(np.diff(signal))
x = d - np.mean(d)

start,end,score = kadane(x)

print("Max deviation interval:",start,end)

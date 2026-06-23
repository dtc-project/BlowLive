import os
import csv
import re
import numpy as np

# ----- Data structure -----
class GFCCData:
    def __init__(self, name, data):
        self.name = name  # string
        self.data = data  # list of list of int64

# ----- Name extraction -----
def extractName(filename):
    match = re.match(r'^[^_]+', os.path.basename(filename))
    if match:
        return match.group(0).lower()
    return ""

# ----- CSV reader -----
def readCSV(filename):
    matrix = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for record in reader:
            row = [int(float(value) * 1e9) for value in record]
            matrix.append(row)
    return matrix

# ----- Local Manhattan distance -----
def localDistanceManhattan(x, y):
    if len(x) != len(y):
        raise ValueError("arrays have different dimensions!")
    return sum(abs(a - b) for a, b in zip(x, y))

# ----- DTW -----
def computeDTW(x, y, unused, distfunc):
    n1, n2 = len(x), len(y)
    if len(x[0]) != len(y[0]):
        raise ValueError("time series elements have different dimensions!")
    INF = (1 << 62)
    dtw = [[INF] * n1 for _ in range(n2)]
    dtw[0][0] = 0
    for j in range(n2):
        for i in range(n1):
            if i > 0 or j > 0:
                cost = distfunc(x[i], y[j])
                candidates = []
                if i > 0:
                    candidates.append(dtw[j][i-1])
                if j > 0:
                    candidates.append(dtw[j-1][i])
                if i > 0 and j > 0:
                    candidates.append(dtw[j-1][i-1])
                dtw[j][i] = cost + min(candidates)
    return dtw[-1][-1]

# ----- Timing utility -----
def avg_over_runs(func, repeat=20, *args):
    results = []
    for _ in range(repeat):
        results.append(func(*args))
    return sum(results)/len(results)

# ----- Load all *_gfcc.csv files -----
def load_gfcc_dataset(datasetDir):
    csvFiles = []
    for root, _, files in os.walk(datasetDir):
        for file in files:
            if "_gfcc.csv" in file:
                csvFiles.append(os.path.join(root, file))
    print("Total matched files:", len(csvFiles))

    gfcc_list = []
    for file in csvFiles:
        name = extractName(file)
        data = readCSV(file)
        gfcc_list.append(GFCCData(name, data))

    gfcc_list.sort(key=lambda x: x.name)
    return gfcc_list

# ----- Compute DTW matrix with averaging -----
def compute_avg_dtw_matrix(gfcc_list):
    n = len(gfcc_list)
    dtw_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            avg_dtw = avg_over_runs(computeDTW, 20, gfcc_list[i].data, gfcc_list[j].data, 0, localDistanceManhattan)
            dtw_matrix[i,j] = avg_dtw
            print(f"DTW[{i},{j}] avg={avg_dtw:.2f}")
    return dtw_matrix

import time

def avg_dtw_time(gfcc_list, subset_size=5, repeat=2):
    """
    Compute the average runtime of DTW computation over `repeat` runs
    using a subset_size x subset_size sample of GFCC data.
    """
    n = min(subset_size, len(gfcc_list))
    total_time = 0
    count = 0

    for i in range(n):
        for j in range(n):
            for _ in range(repeat):
                start = time.time()
                _ = computeDTW(gfcc_list[i].data, gfcc_list[j].data, 0, localDistanceManhattan)
                end = time.time()
                total_time += (end - start)
                print(end-start)
                count += 1

    avg_time = total_time / count
    print(f"Average DTW computation time over {count} runs: {avg_time*1000:.2f} ms")
    return avg_time

# ----- Cosine similarity -----
def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# ----- Read _img.csv embeddings -----
class ImageEmbedding:
    def __init__(self, name, vector):
        self.name = name
        self.vector = vector  # list or np.array


def read_image_embeddings(datasetDir):
    csvFiles = []
    for root, _, files in os.walk(datasetDir):
        for file in files:
            if "_img.csv" in file:
                csvFiles.append(os.path.join(root, file))

    print("Found _img.csv files:", len(csvFiles))

    embeddings = []
    for file in csvFiles:
        name = extractName(file)
        # robust read: assume one-column numeric CSV
        with open(file, "r") as f:
            vector = [float(line.strip()) for line in f if line.strip()]
        embeddings.append(ImageEmbedding(name, vector))

    # sort by name
    embeddings.sort(key=lambda x: x.name)
    return embeddings


def avg_cosine_time(embeddings, subset_size=5, repeat=2):
    """
    Compute the average runtime of cosine similarity computation over `repeat` runs
    using a subset_size x subset_size sample of embeddings.
    """
    n = min(subset_size, len(embeddings))
    total_time = 0
    count = 0

    for i in range(n):
        for j in range(n):
            if i != j:  # skip self-comparison
                for _ in range(repeat):
                    start = time.time()
                    _ = cosine_similarity(embeddings[i].vector, embeddings[j].vector)
                    end = time.time()
                    total_time += (end - start)
                    print(end - start)
                    count += 1

    avg_time = total_time / count
    print(f"Average cosine computation time over {count} runs: {avg_time*1000:.4f} ms")
    return avg_time

# ----- Example usage -----
if __name__ == "__main__":
    datasetDir = "BlowLive_Dataset"
    gfcc_list = load_gfcc_dataset(datasetDir)
    dtw_matrix = avg_dtw_time(gfcc_list)

    img_list = read_image_embeddings(datasetDir)
    cosine_matrix = avg_cosine_time(img_list)
